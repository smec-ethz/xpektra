# Copyright (C) 2025 ETH Zurich (SMEC)
#
# This file is part of xpektra.
#
# xpektra is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# xpektra is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with xpektra.  If not, see <https://www.gnu.org/licenses/>.

"""Contractions spelled so they do not become a batched GEMM on GPU.

Why this module exists
----------------------
``xpektra`` fields are ``(*spatial, *tensor)``: the spatial axes are large
(``N**3`` voxels) and the tensor axes are tiny (2 or 3).  When an einsum sums
over an index shared by both operands, XLA emits a ``dot_general`` and hands the
work to the machinery built for matrix products -- on GPU, a batched GEMM.  That
machinery tiles the operands, stages tiles into on-chip memory and keeps the
arithmetic units saturated, and it pays a fixed setup cost per matrix.  Our
matrices are 3x3.  We pay the setup cost once per voxel to do almost no
arithmetic, and a GEMM is a hard fusion boundary, so every intermediate is
written to main memory and read straight back.  Data movement becomes the whole
cost.

The rule is short: **an einsum becomes a dot if and only if it sums over an
index shared by both operands.**  Free indices are irrelevant.  So
``"...ij,...ji->..."`` and ``"...qj,...qji->...i"`` are dots however little they
look like matrix products, while ``"...qi,...j->...qij"`` and
``"...ij,...kl->...ijkl"`` are not, because nothing is summed.  That is why
``apply_gradient`` and ``apply_symmetric_gradient`` need nothing from this
module and ``apply_divergence`` does.

Writing the same arithmetic as a broadcast multiply followed by a sum gives XLA
nothing to pattern-match.  It stays an ordinary loop, fuses into its neighbours,
and the data is touched once.

How to spot it::

    txt = jax.jit(fn).lower(*args).compile().as_text()
    txt.count(" dot(")                            # any dot_general at all
    txt.count('custom_call_target="__cublas')     # handed to cuBLAS

Note that a contraction of the *operator against itself* (the Laplacian symbol,
the Galerkin ``norm_sq``) is constant-folded away when the scheme is captured as
a closure constant, and is a real dot when the scheme crosses ``jit`` as a
pytree argument -- as it does on the sharded paths.  Check both.

CPU wants the broadcast too
---------------------------
Removing the dot is the mechanism, not the goal, so this was measured rather
than assumed -- and the answer here is *not* the one ``tatva.linalg`` found.
Broadcast wins on CPU as well, by 3-12x, consistently across sizes and dtypes
(one thread, float64 and complex128)::

    N    spec                    dot        broadcast   ratio
    32   ...ij,...ji->...          1.54 ms     0.37 ms   0.24x
    64   ...ij,...ji->...         12.68 ms     2.97 ms   0.23x
    96   ...ij,...ji->...         66.91 ms     9.66 ms   0.14x
    32   ...qj,...qji->...i        3.30 ms     0.96 ms   0.29x
    64   ...qj,...qji->...i       47.93 ms     5.58 ms   0.12x
    96   ...qj,...qji->...i      165.08 ms    17.40 ms   0.11x

The gap widens with ``N``.  tatva's opposite finding -- ``@`` beating broadcast
5-7x on CPU -- is about an *un-batched* ``(p,q) x (q,r)`` product on
element-sized arrays, where the dot is a single small GEMM XLA can fold.  Here
every contraction is already batched over the whole grid with a contracted
dimension of 3 or 6, so the CPU dot path hits the same tiny-batched-GEMM
pathology as the GPU one, just less severely.  The two results do not conflict;
they are different shapes.

So entries are broadcast on both backends by default.  ``cpu_prefers_dot`` keeps
the per-entry switch available (via ``jax.lax.platform_dependent``) because the
question is per-contraction: a spec whose contracted dimension can grow large is
exactly the case GEMM machinery was built for, and would answer differently.
GPU numbers are still outstanding; the structural argument and tatva's GPU
measurements both point the same way, but no entry should be flipped on that
reasoning alone.

Difference from ``tatva.linalg``
--------------------------------
That module solves the same problem at element level and *refuses* a batch axis,
because there batching is ``jax.vmap``'s job and its broadcast form would
materialise the intermediate.  Here the batch is the spatial grid: it is always
present, always leading, and always the large one.  The helpers below therefore
index from the right and place no bound on the leading axes.

The consequence is that these forms depend on fusion.  Read literally,
``"...abcd,...dcef->...abef"`` builds a ``d**6`` product per voxel before
reducing it to ``d**4`` -- at ``N=256`` in complex128 that intermediate would be
~195 GB, against 21 GB for the output.  XLA does not evaluate it literally: it
fuses the multiply into the reduction, so each output value accumulates its
terms in registers and the product is never written anywhere.  Measured, temp
allocation is exactly zero for every entry in the table, in a single fusion.

That is a compiler heuristic rather than a guarantee, so
``test_broadcast_materialises_no_intermediate`` pins it.  It also means an
entry must not be refactored to give the intermediate a second consumer --
that forces materialisation.  A new entry needs the memory check as well as the
dot count; neither shows up at the ``N=32`` used locally.

Adding an entry
---------------
Entries are measured decisions, on both backends, not guesses.  An unrecognised
spec is refused by :func:`contract`; :func:`einsum` falls back to ``jnp.einsum``
so a call site can migrate before its spec has been measured.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["contract", "det", "dyad", "einsum", "inv", "trace", "trans"]


# ---------------------------------------------------------------------------
# Broadcast spellings.  Each one indexes from the right, so any number of
# leading (spatial, quadrature) axes ride along untouched.
# ---------------------------------------------------------------------------


def _bcast_inner1(A: Array, B: Array) -> Array:
    """``...a,...a->...`` -- vector-vector inner product."""
    return jnp.sum(A * B, axis=-1)


def _bcast_inner2(A: Array, B: Array) -> Array:
    """``...ab,...ab->...`` -- contract both axes, aligned.

    The Laplacian symbol ``sum_q sum_i D_qi * Dd_qi`` and the Galerkin
    ``norm_sq``.
    """
    return jnp.sum(A * B, axis=(-2, -1))


def _bcast_inner2_swapped(A: Array, B: Array) -> Array:
    """``...ab,...ba->...`` -- the double dot ``A_ij B_ji``."""
    return jnp.sum(A * jnp.swapaxes(B, -2, -1), axis=(-2, -1))


def _bcast_vec_mat(A: Array, B: Array) -> Array:
    """``...a,...ab->...b`` -- vector against the first axis of a tensor."""
    return jnp.sum(A[..., :, None] * B, axis=-2)


def _bcast_div(A: Array, B: Array) -> Array:
    """``...ab,...abc->...c`` -- ``apply_divergence``: ``Dd_qj sigma_qji``."""
    return jnp.sum(A[..., :, :, None] * B, axis=(-3, -2))


def _bcast_galerkin(A: Array, B: Array) -> Array:
    """``...ab,...acb->...c`` -- Galerkin: ``conj(D_ql) sigma_qil``."""
    return jnp.sum(A[..., :, None, :] * B, axis=(-3, -1))


def _bcast_green(A: Array, B: Array) -> Array:
    """``...abcd,...cd->...ab`` -- Moulinec-Suquet: ``Ghat_khij eps_ij``."""
    return jnp.sum(A * B[..., None, None, :, :], axis=(-2, -1))


def _bcast_mat_vec(A: Array, B: Array) -> Array:
    """``...ab,...b->...a`` -- tensor against a vector."""
    return jnp.sum(A * B[..., None, :], axis=-1)


def _bcast_mat_mat(A: Array, B: Array) -> Array:
    """``...ab,...bc->...ac`` -- tensor-tensor single contraction."""
    return jnp.sum(A[..., :, :, None] * B[..., None, :, :], axis=-2)


def _bcast_mat_rank4(A: Array, B: Array) -> Array:
    """``...ab,...bcde->...acde`` -- rank-2 against a rank-4's first axis."""
    return jnp.sum(A[..., :, :, None, None, None] * B[..., None, :, :, :, :], axis=-4)


def _bcast_rank4_mat(A: Array, B: Array) -> Array:
    """``...abcd,...de->...abce`` -- rank-4 against a rank-2."""
    return jnp.sum(A[..., :, :, :, :, None] * B[..., None, None, None, :, :], axis=-2)


def _bcast_rank4_ddot_mat(A: Array, B: Array) -> Array:
    """``...abcd,...dc->...ab`` -- the tangent applied to a strain.

    ``B``'s contracted axes are ``(d, c)`` where ``A``'s are ``(c, d)``, so it is
    transposed first rather than the product being re-indexed.
    """
    Bs = jnp.swapaxes(B, -2, -1)
    return jnp.sum(A * Bs[..., None, None, :, :], axis=(-2, -1))


def _bcast_rank4_ddot_rank4(A: Array, B: Array) -> Array:
    """``...abcd,...dcef->...abef`` -- rank-4 double dot rank-4.

    The widest entry in the table: read literally the product is ``d**6`` per
    voxel (729 at ``d = 3``) before reducing to ``d**4``.  XLA fuses it into the
    reduction so nothing is materialised -- measured temp allocation is zero --
    but that is what makes this spelling viable at all.  See the module
    docstring.
    """
    Bs = jnp.swapaxes(B, -4, -3)
    return jnp.sum(
        A[..., :, :, :, :, None, None] * Bs[..., None, None, :, :, :, :], axis=(-4, -3)
    )


class _Contraction(NamedTuple):
    """How one contraction is spelled, and the shapes it accepts.

    Attributes:
        broadcast: the fusable spelling.
        n_axes: trailing tensor axes consumed from ``A`` and ``B``.  Leading
            axes are unconstrained and must broadcast against each other.
        cpu_prefers_dot: whether CPU was *measured* to be faster with the
            ``dot``.  No current entry sets it -- see the module docstring for
            the numbers -- but the switch is kept because the question is
            per-contraction, and an entry whose contracted dimension can grow
            large would answer it differently.
    """

    broadcast: Callable[[Array, Array], Array]
    n_axes: tuple[int, int]
    cpu_prefers_dot: bool = False


# Canonical spec -> implementation.  Keys are canonicalised (see
# ``_canonicalise``), so a call site may use letters that carry meaning:
# ``"...qj,...qji->...i"`` and ``"...ab,...abc->...c"`` are the same entry.
_KNOWN_CONTRACTIONS: dict[str, _Contraction] = {
    "...a,...a->...": _Contraction(_bcast_inner1, (1, 1)),
    "...ab,...ab->...": _Contraction(_bcast_inner2, (2, 2)),
    "...ab,...ba->...": _Contraction(_bcast_inner2_swapped, (2, 2)),
    "...a,...ab->...b": _Contraction(_bcast_vec_mat, (1, 2)),
    "...ab,...abc->...c": _Contraction(_bcast_div, (2, 3)),
    "...ab,...acb->...c": _Contraction(_bcast_galerkin, (2, 3)),
    "...abcd,...cd->...ab": _Contraction(_bcast_green, (4, 2)),
    "...ab,...b->...a": _Contraction(_bcast_mat_vec, (2, 1)),
    "...ab,...bc->...ac": _Contraction(_bcast_mat_mat, (2, 2)),
    "...ab,...bcde->...acde": _Contraction(_bcast_mat_rank4, (2, 4)),
    "...abcd,...de->...abce": _Contraction(_bcast_rank4_mat, (4, 2)),
    "...abcd,...dc->...ab": _Contraction(_bcast_rank4_ddot_mat, (4, 2)),
    "...abcd,...dcef->...abef": _Contraction(_bcast_rank4_ddot_rank4, (4, 4)),
}


def _tokenize(term: str) -> list[str]:
    """Split a spec term into index letters, with ``...`` as a single token."""
    tokens, i = [], 0
    while i < len(term):
        if term.startswith("...", i):
            tokens.append("...")
            i += 3
        else:
            tokens.append(term[i])
            i += 1
    return tokens


def _canonicalise(spec: str) -> str:
    """Rename index letters in order of first appearance.

    ``"...qj,...qji->...i"`` becomes ``"...ab,...abc->...c"``, so the table holds
    one entry per *contraction* rather than one per spelling.
    """
    cleaned = spec.replace(" ", "")
    if cleaned.count("->") != 1 or cleaned.count(",") != 1:
        raise ValueError(
            f"xpektra.linalg could not parse the spec {spec!r}. Expected exactly "
            "one ',' and one '->', as in '...ij,...ji->...'."
        )
    lhs, out = cleaned.split("->")
    a_term, b_term = lhs.split(",")

    renaming: dict[str, str] = {}

    def rename(term: str) -> str:
        letters = []
        for token in _tokenize(term):
            if token == "...":
                letters.append(token)
                continue
            if not token.isalpha():
                raise ValueError(
                    f"xpektra.linalg could not parse the spec {spec!r}: "
                    f"{token!r} is not an index letter."
                )
            if token not in renaming:
                renaming[token] = chr(ord("a") + len(renaming))
            letters.append(renaming[token])
        return "".join(letters)

    return f"{rename(a_term)},{rename(b_term)}->{rename(out)}"


def contract(spec: str, A: Array, B: Array) -> Array:
    """Contract two fields, named by an einsum-style spec.

        div  = linalg.contract("...qj,...qji->...i", Dd, sigma)
        ddot = linalg.contract("...ij,...ji->...",   A,  B)

    Index letters are free; they are canonicalised by order of first appearance.
    Leading axes (spatial, quadrature) are unconstrained -- only the trailing
    tensor axes named by the spec are consumed.

    This is the measured broadcast spelling, which does not lower to a ``dot``.
    An entry marked ``cpu_prefers_dot`` uses ``jnp.einsum`` on CPU instead; none
    currently does.  See the module docstring for the measurements.

    Args:
        spec: einsum-style contraction, e.g. ``"...ij,...ji->..."``.  Must name
            one of the measured contractions; letters are arbitrary.
        A: first operand.
        B: second operand.

    Returns:
        The contraction, with axes in the order the spec's output term gives.

    Raises:
        ValueError: if the spec is unparseable, names a contraction with no
            measured implementation, or disagrees with the operand shapes.
    """
    canonical = _canonicalise(spec)
    entry = _KNOWN_CONTRACTIONS.get(canonical)

    if entry is None:
        supported = ", ".join(repr(s) for s in _KNOWN_CONTRACTIONS)
        raise ValueError(
            f"xpektra.linalg.contract has no implementation for {spec!r} "
            f"(canonically {canonical!r}). Supported contractions are {supported}. "
            "This is deliberate rather than a gap: each entry is a measured "
            "decision, on both backends, about how to spell the contraction so it "
            "does not lower to a `dot`. Add an entry with its measurement rather "
            "than reaching for jnp.einsum -- see the module docstring."
        )

    n_a, n_b = entry.n_axes
    if A.ndim < n_a or B.ndim < n_b:
        raise ValueError(
            f"xpektra.linalg.contract({spec!r}, ...) needs at least {n_a} and "
            f"{n_b} trailing tensor axes, got shapes {A.shape} and {B.shape}."
        )

    if not entry.cpu_prefers_dot:
        return entry.broadcast(A, B)

    def _dot(a: Array, b: Array) -> Array:
        return jnp.einsum(spec, a, b)

    return jax.lax.platform_dependent(A, B, cpu=_dot, default=entry.broadcast)


def einsum(spec: Any, *operands: Any, **kwargs: Any) -> Array:
    """Drop-in replacement for ``jnp.einsum`` that routes measured contractions.

    A two-operand string spec naming one of the measured contractions takes the
    :func:`contract` path; everything else -- three or more operands, an
    implicit output, the interleaved calling convention, and any contraction
    that has not been measured -- goes to ``jnp.einsum`` unchanged.  That
    fallback is what makes migration safe: a call site can switch to this
    function before its spec is in the table, with no change in behaviour.

    The :func:`contract` path ignores keyword arguments such as ``precision``
    and ``preferred_element_type``, so call ``jnp.einsum`` directly when one of
    those has to hold.

    Args:
        spec: einsum-style subscripts, e.g. ``"...ij,...ji->..."``.
        operands: the tensors to contract.
        kwargs: forwarded verbatim to ``jnp.einsum`` on every path reaching it.

    Returns:
        The contraction.
    """
    if not isinstance(spec, str) or kwargs:
        return jnp.einsum(spec, *operands, **kwargs)

    try:
        canonical = _canonicalise(spec)
    except ValueError:
        # a spec `contract` cannot name: implicit output, one operand, or more
        # than two. Nothing to look up.
        return jnp.einsum(spec, *operands, **kwargs)

    if canonical not in _KNOWN_CONTRACTIONS or len(operands) != 2:
        return jnp.einsum(spec, *operands, **kwargs)

    return contract(spec, *operands)


# ---------------------------------------------------------------------------
# Pointwise tensor algebra on ``(*leading, *tensor)`` fields.
#
# These take no rank argument where the operation fixes it (``trace``, ``trans``
# act on the last two axes) and require one where it does not (``dyad``).
# Nothing here infers a tensor rank from ``ndim``: that inference is what made
# the old ``TensorOperator`` unable to tell a node field ``(*spatial, dim)``
# from a centre field ``(*spatial, n_quads)``, and it silently dispatched the
# wrong rule rather than failing.
# ---------------------------------------------------------------------------


def trace(A: Array) -> Array:
    """Trace over the last two axes: ``A_...ii``.

    Deliberately ``jnp.trace`` rather than ``einsum("...ii->...")``: a repeated
    index *within one operand* has no XLA primitive, so it is emulated with a
    mask-and-select.

    Unsharded, both spellings lower to the same ``iota``/``compare``/``select``,
    so comparing HLO will not tell them apart.  They diverge under **explicit**
    sharding, where the einsum builds its mask replicated and raises
    ``ShardingTypeError: select `which` must be scalar or have the same sharding
    as cases``, while ``jnp.trace`` produces a correctly sharded one.  Verified
    on jax 0.10.0; ``test_trace_survives_explicit_sharding`` pins it.

    Args:
        A: field of square blocks, shape ``(*leading, d, d)``.

    Returns:
        Shape ``(*leading,)``.
    """
    return jnp.trace(A, axis1=-2, axis2=-1)


def trans(A: Array) -> Array:
    """Transpose the last two axes: ``A_...ij -> A_...ji``.

    Args:
        A: field of blocks, shape ``(*leading, m, n)``.

    Returns:
        Shape ``(*leading, n, m)``.
    """
    return jnp.swapaxes(A, -1, -2)


def dyad(A: Array, B: Array, rank: int) -> Array:
    """Outer product of the trailing ``rank`` axes of each operand.

    ``rank`` is required and has no default.  Unlike ``trace`` and ``trans``,
    the operation does not fix how many trailing axes are tensor axes, and
    guessing from ``ndim`` cannot work: ``(*spatial, d)`` and ``(*spatial, q)``
    are the same shape.  Equivalent to the explicit einsum, which is also
    contraction-free and so never lowers to a ``dot``::

        dyad(a, b, rank=1)  ==  jnp.einsum("...i,...j->...ij",     a, b)
        dyad(A, B, rank=2)  ==  jnp.einsum("...ij,...kl->...ijkl", A, B)

    Args:
        A: field with ``rank`` trailing tensor axes.
        B: field with ``rank`` trailing tensor axes.
        rank: number of trailing tensor axes on each operand; 1 or 2.

    Returns:
        Field with ``2 * rank`` trailing tensor axes.
    """
    if rank == 1:
        return jnp.einsum("...i,...j->...ij", A, B)
    if rank == 2:
        return jnp.einsum("...ij,...kl->...ijkl", A, B)
    raise ValueError(
        f"xpektra.linalg.dyad supports rank 1 and 2, got {rank}. Write the "
        "outer product as an explicit jnp.einsum; it contracts nothing, so it "
        "cannot lower to a dot."
    )


def det(A: Array) -> Array:
    """Pointwise determinant of 1x1, 2x2 or 3x3 blocks.

    Closed form, selected on the static block size, so no batched LU is
    dispatched -- ``jnp.linalg.det`` runs a pivoted factorisation per block,
    which is right for one large matrix and ruinous over a grid of tiny ones.

    Companion to :func:`inv`: thresholding on ``abs(det)`` is how a caller
    identifies the blocks ``inv`` cannot handle.

    Args:
        A: field of square blocks, shape ``(*leading, d, d)`` with d in 1, 2, 3.

    Returns:
        Shape ``(*leading,)``.
    """
    d = _block_size(A, "det")
    if d == 1:
        return A[..., 0, 0]
    if d == 2:
        return A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    # scalar triple product of the columns
    a1, a2, a3 = A[..., :, 0], A[..., :, 1], A[..., :, 2]
    return jnp.sum(a1 * jnp.cross(a2, a3), axis=-1)


def inv(A: Array) -> Array:
    """Pointwise inverse of 1x1, 2x2 or 3x3 blocks.

    Each block is inverted by a closed-form expression, for the same reason as
    :func:`det`.  The formulas are unguarded: a singular block yields
    ``inf``/``nan`` rather than raising.  Null-space handling belongs to the
    caller, which knows which modes are expected to be singular (the ``xi = 0``
    mode of a Green's operator, the Nyquist corner of a reduced-integration
    stencil).  Use :func:`det` to mask them beforehand.

    Args:
        A: field of square blocks, shape ``(*leading, d, d)`` with d in 1, 2, 3.

    Returns:
        Same shape as ``A``.
    """
    d = _block_size(A, "inv")
    if d == 1:
        return 1.0 / A
    if d == 2:
        a, b = A[..., 0, 0], A[..., 0, 1]
        c, e = A[..., 1, 0], A[..., 1, 1]
        adj = jnp.stack(
            [jnp.stack([e, -b], axis=-1), jnp.stack([-c, a], axis=-1)], axis=-2
        )
        return adj / (a * e - b * c)[..., None, None]
    # Reciprocal-basis form: the rows of ``det * A^-1`` are the cross products of
    # the columns of A, which builds the adjugate already transposed.
    a1, a2, a3 = A[..., :, 0], A[..., :, 1], A[..., :, 2]
    c1, c2, c3 = jnp.cross(a2, a3), jnp.cross(a3, a1), jnp.cross(a1, a2)
    return jnp.stack([c1, c2, c3], axis=-2) / jnp.sum(a1 * c1, axis=-1)[..., None, None]


def _block_size(A: Array, name: str) -> int:
    """Validates a field of square blocks and returns the block size."""
    if A.ndim < 2:
        raise ValueError(
            f"xpektra.linalg.{name} needs a field of square blocks "
            f"(*leading, d, d), got shape {A.shape}."
        )
    rows, cols = A.shape[-2], A.shape[-1]
    if rows != cols:
        raise ValueError(
            f"xpektra.linalg.{name} requires square blocks, got ({rows}, {cols}) "
            f"from shape {A.shape}."
        )
    if cols not in (1, 2, 3):
        raise ValueError(
            f"xpektra.linalg.{name} has a closed form for blocks of size 1, 2 "
            f"and 3, got {cols}."
        )
    return cols
