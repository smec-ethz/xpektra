"""``xpektra.linalg``: the broadcast spellings are correct, and dot-free.

The point of the module is that a contraction can be spelled two ways that agree
numerically and differ enormously in cost on GPU.  These tests pin both halves:
the broadcast form computes the same thing as the einsum it replaces, and it
compiles without a ``dot_general``.

The dot-free assertion is made against the broadcast implementation directly,
not against ``contract``.  On CPU ``contract`` deliberately *is* the einsum --
``dot`` is the faster spelling there and XLA fuses it away -- so asserting "no
dot" on the public entry point would assert the opposite of what we want on the
backend this suite runs on.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import jax

jax.config.update("jax_enable_x64", True)

import re

import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from xpektra.linalg import _KNOWN_CONTRACTIONS, _canonicalise, contract, einsum

D, Q = 3, 2
LEAD = (4, 5)

# (spec as a call site would write it, A tensor axes, B tensor axes).
# Every entry in ``_KNOWN_CONTRACTIONS`` must appear here -- see
# ``test_every_table_entry_is_covered``.
SPECS = [
    # scheme / projection operators
    ("...qj,...qji->...i", (Q, D), (Q, D, D)),  # apply_divergence
    ("...qi,...qi->...", (Q, D), (Q, D)),  # apply_laplacian, Galerkin norm_sq
    ("...ql,...qil->...i", (Q, D), (Q, D, D)),  # GalerkinProjection
    ("...khij,...ij->...kh", (D, D, D, D), (D, D)),  # MoulinecSuquetProjection
    # TensorOperator.dot
    ("...i,...i->...", (D,), (D,)),
    ("...i,...ij->...j", (D,), (D, D)),
    ("...ij,...j->...i", (D, D), (D,)),
    ("...ik,...kj->...ij", (D, D), (D, D)),
    ("...ik,...klmn->...ilmn", (D, D), (D, D, D, D)),
    ("...ijkl,...lm->...ijkm", (D, D, D, D), (D, D)),
    # TensorOperator.ddot
    ("...ij,...ji->...", (D, D), (D, D)),
    ("...ijkl,...lk->...ij", (D, D, D, D), (D, D)),
    ("...ijkl,...lkmn->...ijmn", (D, D, D, D), (D, D, D, D)),
]
IDS = [s for s, _, _ in SPECS]


def test_every_table_entry_is_covered():
    """No entry may be added to the table without a test exercising it."""
    covered = {_canonicalise(s) for s, _, _ in SPECS}
    missing = set(_KNOWN_CONTRACTIONS) - covered
    assert not missing, f"table entries with no test: {sorted(missing)}"


def _operands(spec_a, spec_b, seed=0):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    A = jax.random.normal(k1, LEAD + spec_a) + 1j * jax.random.normal(k1, LEAD + spec_a)
    B = jax.random.normal(k2, LEAD + spec_b) + 1j * jax.random.normal(k2, LEAD + spec_b)
    return A, B


def _count_dots(fn, *args) -> int:
    return jax.jit(fn).lower(*args).compile().as_text().count(" dot(")


def _opcodes(txt: str) -> set[str]:
    """HLO instruction opcodes only.

    The compiled text also carries ``metadata=`` with Python source paths, so a
    plain substring search over it gives false positives -- "lu" matches
    "pluggy".
    """
    return set(re.findall(r"= \S+ ([a-z-]+)\(", txt))


@pytest.mark.parametrize(("spec", "sa", "sb"), SPECS, ids=IDS)
def test_broadcast_matches_einsum(spec, sa, sb):
    """The measured spelling computes what its spec says."""
    A, B = _operands(sa, sb)
    impl = _KNOWN_CONTRACTIONS[_canonicalise(spec)].broadcast
    expected = jnp.einsum(spec, A, B)
    got = impl(A, B)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=1e-12)


@pytest.mark.parametrize(("spec", "sa", "sb"), SPECS, ids=IDS)
def test_broadcast_emits_no_dot(spec, sa, sb):
    """The broadcast spelling does not lower to a ``dot_general``.

    This is the regression that matters: reverting one of these to
    ``jnp.einsum`` leaves every numerical test passing and silently reinstates a
    batched GEMM on GPU.
    """
    A, B = _operands(sa, sb)
    impl = _KNOWN_CONTRACTIONS[_canonicalise(spec)].broadcast
    assert _count_dots(impl, A, B) == 0, (
        f"broadcast spelling of {spec!r} lowered to a dot_general"
    )


@pytest.mark.parametrize(("spec", "sa", "sb"), SPECS, ids=IDS)
def test_broadcast_materialises_no_intermediate(spec, sa, sb):
    """The broadcast product is fused into the reduce, never written out.

    This is the other half of the bargain.  ``...ijkl,...lkmn->...ijmn`` reads as
    a ``d**6`` per-voxel product reduced to ``d**4``; if XLA evaluated that
    literally the intermediate would be 9x the output -- ~195 GB at N=256 in
    complex128, against 21 GB for the output.  It does not: measured temp
    allocation is exactly zero for every entry, in a single fusion.

    Fusion is a compiler heuristic rather than a guarantee, so this pins it.  A
    failure here means an XLA change (or a refactor that gives the intermediate a
    second consumer) turned a working kernel into an out-of-memory one -- a
    failure mode no numerical test would catch.
    """
    A, B = _operands(sa, sb)
    impl = _KNOWN_CONTRACTIONS[_canonicalise(spec)].broadcast
    analysis = jax.jit(impl).lower(A, B).compile().memory_analysis()
    assert analysis.temp_size_in_bytes <= analysis.output_size_in_bytes, (
        f"{spec!r} materialised a {analysis.temp_size_in_bytes} B intermediate "
        f"against a {analysis.output_size_in_bytes} B output; the broadcast "
        "product is no longer being fused into the reduction"
    )


@pytest.mark.parametrize(("spec", "sa", "sb"), SPECS, ids=IDS)
def test_einsum_reference_does_emit_a_dot(spec, sa, sb):
    """Every spec in the table really is a contraction XLA turns into a dot.

    Guards the table against entries that never needed to be there -- if this
    fails, the spec sums over nothing shared and belongs in ``jnp.einsum``.
    """
    A, B = _operands(sa, sb)
    assert _count_dots(lambda a, b: jnp.einsum(spec, a, b), A, B) > 0, (
        f"{spec!r} does not lower to a dot; it does not need an entry"
    )


@pytest.mark.parametrize(("spec", "sa", "sb"), SPECS, ids=IDS)
def test_contract_matches_einsum_on_this_backend(spec, sa, sb):
    """Whichever branch this backend selects, the answer is unchanged."""
    A, B = _operands(sa, sb)
    np.testing.assert_allclose(contract(spec, A, B), jnp.einsum(spec, A, B), atol=1e-12)


@pytest.mark.parametrize(("spec", "sa", "sb"), SPECS, ids=IDS)
def test_contract_emits_no_dot(spec, sa, sb):
    """No entry is marked ``cpu_prefers_dot``, so ``contract`` is dot-free here.

    Measured: broadcast beats the dot on CPU by 3-12x for every spec in the
    table, so unlike ``tatva.linalg`` there is no backend split to preserve. If
    an entry ever sets ``cpu_prefers_dot``, exclude it here rather than
    weakening the assertion.
    """
    A, B = _operands(sa, sb)
    assert _count_dots(lambda a, b: contract(spec, a, b), A, B) == 0


def test_letters_are_free():
    """A call site may name indices meaningfully; the table keys on the shape."""
    A, B = _operands((Q, D), (Q, D, D))
    np.testing.assert_allclose(
        contract("...qj,...qji->...i", A, B),
        contract("...ab,...abc->...c", A, B),
        atol=1e-14,
    )


def test_leading_axes_are_unconstrained():
    """Any number of spatial/quadrature axes ride along untouched.

    This is where xpektra departs from ``tatva.linalg``, which refuses a batch
    axis outright.
    """
    for lead in [(), (7,), (4, 5), (2, 3, 4)]:
        k1, k2 = jax.random.split(jax.random.PRNGKey(1))
        A = jax.random.normal(k1, lead + (D, D))
        B = jax.random.normal(k2, lead + (D, D))
        np.testing.assert_allclose(
            contract("...ij,...ji->...", A, B),
            jnp.einsum("...ij,...ji->...", A, B),
            atol=1e-12,
        )


def test_unknown_spec_is_refused_by_contract():
    A, B = _operands((D,), (D,))
    with pytest.raises(ValueError, match="no implementation"):
        contract("...i,...j->...ij", A, B)


def test_einsum_falls_back_for_unmeasured_specs():
    """The shim is a safe drop-in: an unmeasured spec still works."""
    A, B = _operands((D,), (D,))
    np.testing.assert_allclose(
        einsum("...i,...j->...ij", A, B),
        jnp.einsum("...i,...j->...ij", A, B),
        atol=1e-14,
    )


def test_einsum_routes_measured_specs_through_contract():
    A, B = _operands((D, D), (D, D))
    np.testing.assert_allclose(
        einsum("...ij,...ji->...", A, B),
        contract("...ij,...ji->...", A, B),
        atol=1e-14,
    )


def test_einsum_falls_back_for_other_calling_conventions():
    """Three operands, interleaved form and kwargs all bypass the table."""
    A, B = _operands((D, D), (D, D))
    C = jnp.eye(D)
    np.testing.assert_allclose(
        einsum("...ij,...jk,kl->...il", A, B, C),
        jnp.einsum("...ij,...jk,kl->...il", A, B, C),
        atol=1e-12,
    )
    # kwargs must reach jnp.einsum, so they cannot take the contract path
    np.testing.assert_allclose(
        einsum("...ij,...ji->...", A, B, precision="highest"),
        jnp.einsum("...ij,...ji->...", A, B, precision="highest"),
        atol=1e-12,
    )


def test_shape_mismatch_is_reported():
    A = jnp.zeros((4, D))
    B = jnp.zeros((4, D))
    with pytest.raises(ValueError, match="trailing tensor axes"):
        contract("...qj,...qji->...i", A, B)


# ---------------------------------------------------------------------------
# Pointwise algebra: trace, trans, dyad, det, inv
# ---------------------------------------------------------------------------

from xpektra.linalg import det, dyad, inv, trace, trans

LEAD3 = (4, 5)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_trace_matches_jnp(d):
    A = jax.random.normal(jax.random.PRNGKey(0), LEAD3 + (d, d))
    np.testing.assert_allclose(trace(A), jnp.trace(A, axis1=-2, axis2=-1), atol=1e-14)


def test_trace_survives_explicit_sharding():
    """``trace`` must stay ``jnp.trace``; ``einsum("...ii->...")`` breaks here.

    Both spellings lower to the same iota/compare/select mask when nothing is
    sharded, so an HLO comparison cannot tell them apart.  Under *explicit*
    sharding they diverge: the einsum builds its mask replicated and fails with
    ``ShardingTypeError: select `which` must be scalar or have the same sharding
    as cases``, while ``jnp.trace`` produces a correctly sharded one.

    So this test is the only thing standing between the implementation and a
    plausible-looking "simplification" to einsum.
    """
    if jax.device_count() < 2:
        pytest.skip("needs >= 2 devices; see XLA_FLAGS at the top of this module")

    mesh = jax.make_mesh((2,), ("x",), axis_types=(jax.sharding.AxisType.Explicit,))
    with jax.sharding.set_mesh(mesh):
        A = jax.device_put(
            jax.random.normal(jax.random.PRNGKey(0), (4, 4, D, D)),
            jax.NamedSharding(mesh, P("x", None, None, None)),
        )
        out = jax.jit(trace)(A)
        np.testing.assert_allclose(
            np.asarray(out), np.trace(np.asarray(A), axis1=-2, axis2=-1), atol=1e-14
        )


def test_trans_swaps_last_two_axes():
    A = jax.random.normal(jax.random.PRNGKey(0), LEAD3 + (2, 3))
    assert trans(A).shape == LEAD3 + (3, 2)
    np.testing.assert_allclose(trans(trans(A)), A, atol=1e-14)


@pytest.mark.parametrize(
    ("rank", "spec"),
    [(1, "...i,...j->...ij"), (2, "...ij,...kl->...ijkl")],
)
def test_dyad_matches_einsum(rank, spec):
    shape = LEAD3 + (D,) * rank
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    A, B = jax.random.normal(k1, shape), jax.random.normal(k2, shape)
    np.testing.assert_allclose(
        dyad(A, B, rank=rank), jnp.einsum(spec, A, B), atol=1e-14
    )


def test_dyad_contracts_nothing_so_it_never_dots():
    A = jax.random.normal(jax.random.PRNGKey(0), LEAD3 + (D, D))
    assert _count_dots(lambda a: dyad(a, a, rank=2), A) == 0


def test_dyad_requires_a_rank():
    A = jax.random.normal(jax.random.PRNGKey(0), LEAD3 + (D,))
    with pytest.raises(TypeError):
        dyad(A, A)  # rank is required: (*spatial, d) and (*spatial, q) look alike
    with pytest.raises(ValueError, match="supports rank 1 and 2"):
        dyad(A, A, rank=3)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_det_matches_jnp(d):
    A = jnp.eye(d) + 0.3 * jax.random.normal(jax.random.PRNGKey(1), LEAD3 + (d, d))
    np.testing.assert_allclose(det(A), jnp.linalg.det(A), rtol=1e-12)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_inv_is_a_left_and_right_inverse(d):
    A = jnp.eye(d) + 0.3 * jax.random.normal(jax.random.PRNGKey(2), LEAD3 + (d, d))
    eye = jnp.broadcast_to(jnp.eye(d), LEAD3 + (d, d))
    np.testing.assert_allclose(
        contract("...ik,...kj->...ij", A, inv(A)), eye, atol=1e-11
    )
    np.testing.assert_allclose(
        contract("...ik,...kj->...ij", inv(A), A), eye, atol=1e-11
    )


@pytest.mark.parametrize("d", [1, 2, 3])
def test_inv_avoids_the_lapack_path(d):
    """Closed form, not a batched factorisation.

    ``jnp.linalg.inv`` lowers to LAPACK ``custom-call``s -- 3 of them here --
    which is right for one large matrix and ruinous over a grid of tiny blocks.
    Counted as an HLO opcode rather than matched as a substring: the compiled
    text embeds Python source paths, and "lu" is a substring of "pluggy".
    """
    A = jnp.eye(d) + 0.3 * jax.random.normal(jax.random.PRNGKey(2), LEAD3 + (d, d))
    opcodes = _opcodes(jax.jit(inv).lower(A).compile().as_text())
    assert "custom-call" not in opcodes, (
        f"inv on {d}x{d} blocks is dispatching a LAPACK custom-call"
    )
    # the reference really does, so the assertion above is not vacuous
    assert "custom-call" in _opcodes(
        jax.jit(jnp.linalg.inv).lower(A).compile().as_text()
    )


@pytest.mark.parametrize("fn", [det, inv])
def test_block_shape_is_validated(fn):
    with pytest.raises(ValueError, match="square blocks"):
        fn(jnp.zeros(LEAD3 + (2, 3)))
    with pytest.raises(ValueError, match="closed form"):
        fn(jnp.zeros(LEAD3 + (4, 4)))
