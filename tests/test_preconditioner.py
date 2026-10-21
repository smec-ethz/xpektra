"""``make_generic_preconditioner`` recovers the inverse symbol of a reference operator.

The construction assumes only that the reference operator is linear and
translation invariant -- not that it is self-adjoint.  These tests exercise both
a non-singular reference (where ``M^-1 K`` is exactly the identity) and a
singular one (where the null modes must be projected out rather than inverted).
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from xpektra import (
    Field,
    SolutionSpace,
    SpectralSpace,
    adapt_green_preconditioner,
    linalg,
)
from xpektra.preconditioner import (
    make_generic_preconditioner,
    make_isotropic_preconditioner,
)
from xpektra.scheme import (
    FourierScheme,
    Hex1RScheme,
    Quad1RScheme,
    QuadFullScheme,
    Tetra2Scheme,
)
from xpektra.spectral_operator import SpectralOperator
from xpektra.transform import FFTTransform

N = 9

SCHEMES = [(Quad1RScheme, 2), (Hex1RScheme, 3), (FourierScheme, 2)]
IDS = [f"{c.__name__}-{d}d" for c, d in SCHEMES]


def _operator(cls, dim):
    space = SpectralSpace(
        lengths=(1.0,) * dim, shape=(N,) * dim, transform=FFTTransform(dim=dim)
    )
    return SpectralOperator(scheme=cls(space=space), space=space), space


def _solution_space(space, dim=1):
    return SolutionSpace(space, u=Field(shape=() if dim == 1 else (dim,)))


def _helmholtz(op, space, dim):
    """``u - laplacian(u)`` on a scalar field: linear, shift invariant, no null mode.

    The mass term is what makes it invertible everywhere, so ``M^-1 K`` should be
    the identity to round-off with nothing to project out.
    """

    def apply(u_flat):
        u = u_flat.reshape(space.shape)
        return (u - op.laplacian(u)).reshape(-1)

    return apply


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_inverts_a_nonsingular_reference(cls, dim):
    """``M^-1 K v == v`` when the reference has no null modes."""
    op, space = _operator(cls, dim)
    K = _helmholtz(op, space, dim)
    M_inv = make_generic_preconditioner(K, _solution_space(space, 1))

    v = jax.random.normal(jax.random.PRNGKey(0), (N**dim,))
    np.testing.assert_allclose(M_inv(K(v)), v, atol=1e-10)


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_null_modes_are_projected_not_inverted(cls, dim):
    """A singular reference must give 0 on its null space, not ``inf``.

    The Laplacian annihilates a constant field.  Its symbol there is round-off
    rather than exactly zero, so an ``== 0`` guard would let it through and the
    closed-form inverse would return ~1e29.
    """
    op, space = _operator(cls, dim)

    def K(u_flat):
        return (-op.laplacian(u_flat.reshape(space.shape))).reshape(-1)

    M_inv = make_generic_preconditioner(K, _solution_space(space, 1))

    assert jnp.all(jnp.isfinite(M_inv.G_hat)), "null mode produced a non-finite inverse"

    constant = jnp.ones((N**dim,))
    np.testing.assert_allclose(M_inv(constant), 0.0, atol=1e-12)


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_is_a_left_inverse_off_the_null_space(cls, dim):
    """``M^-1 K`` is the identity on the range, i.e. ``M^-1 K M^-1 == M^-1``."""
    op, space = _operator(cls, dim)

    def K(u_flat):
        return (-op.laplacian(u_flat.reshape(space.shape))).reshape(-1)

    M_inv = make_generic_preconditioner(K, _solution_space(space, 1))

    r = jax.random.normal(jax.random.PRNGKey(1), (N**dim,))
    z = M_inv(r)
    np.testing.assert_allclose(M_inv(K(z)), z, atol=1e-10)


def test_handles_a_vector_field_reference():
    """``d = ndim``: the symbol is a full ``(d, d)`` block per mode."""
    dim = 2
    op, space = _operator(Quad1RScheme, dim)
    n_u = N**dim * dim

    def K(u_flat):
        u = u_flat.reshape(space.shape + (dim,))
        return (u - op.laplacian(u)).reshape(-1)

    M_inv = make_generic_preconditioner(K, _solution_space(space, dim))
    assert M_inv.G_hat.shape == space.shape + (dim, dim)

    v = jax.random.normal(jax.random.PRNGKey(2), (n_u,))
    np.testing.assert_allclose(M_inv(K(v)), v, atol=1e-10)


def test_makes_no_symmetry_assumption():
    """A non-self-adjoint reference must invert correctly too.

    This is the property non-associated plasticity depends on: its tangent is
    not symmetric, so a construction that assumed self-adjointness would be
    silently wrong rather than merely slower.
    """
    dim = 2
    op, space = _operator(Quad1RScheme, dim)

    def K(u_flat):
        """``u - laplacian(u) + c . grad(u)``: the advection term breaks symmetry."""
        u = u_flat.reshape(space.shape)
        g = op.grad(u)  # (*spatial, n_quads, dim)
        drift = g[..., 0, 0] + 0.5 * g[..., 0, 1]
        return (u - op.laplacian(u) + 0.7 * drift).reshape(-1)

    M_inv = make_generic_preconditioner(K, _solution_space(space, 1))

    # the symbol really is non-Hermitian, so the test is not vacuous
    K_hat_asym = jnp.max(jnp.abs(M_inv.G_hat - jnp.conj(M_inv.G_hat)))
    assert float(K_hat_asym) > 1e-6, "reference came out Hermitian; test is vacuous"

    v = jax.random.normal(jax.random.PRNGKey(3), (N**dim,))
    np.testing.assert_allclose(M_inv(K(v)), v, atol=1e-10)


def test_treedef_is_stable_across_rebuilds():
    """Rebuilding for a new reference state must not force a recompile.

    `G_hat` is the only dynamic field, so two builds differing in data alone have
    identical treedefs -- that is what lets a solver hold the preconditioner as a
    pytree child and swap it between increments.
    """
    dim = 2
    op, space = _operator(Quad1RScheme, dim)

    def make(scale):
        def K(u_flat):
            u = u_flat.reshape(space.shape)
            return (scale * u - op.laplacian(u)).reshape(-1)

        return make_generic_preconditioner(K, _solution_space(space, 1))

    a, b = make(1.0), make(2.5)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    assert not np.allclose(a.G_hat, b.G_hat), "different references gave identical data"


def test_survives_a_jit_boundary():
    """The stored symbol crosses `jit` as a pytree child."""
    dim = 2
    op, space = _operator(Quad1RScheme, dim)
    K = _helmholtz(op, space, dim)
    M_inv = make_generic_preconditioner(K, _solution_space(space, 1))

    r = jax.random.normal(jax.random.PRNGKey(4), (N**dim,))
    out = jax.jit(lambda m, x: m(x))(M_inv, r)
    np.testing.assert_allclose(out, M_inv(r), atol=1e-14)


# ---------------------------------------------------------------------------
# The isotropic (matrix-free) preconditioner
# ---------------------------------------------------------------------------

ISO_SCHEMES = [(Quad1RScheme, 2), (Hex1RScheme, 3)]
ISO_IDS = [f"{c.__name__}-{d}d" for c, d in ISO_SCHEMES]

LAM0, MU0 = 1.3, 0.7


def _isotropic_reference(op, space, dim):
    """``K = D^T C0 D`` for a homogeneous isotropic material, via the energy."""
    u_shape = space.shape + (dim,)

    def energy(u_flat):
        eps = op.sym_grad(u_flat.reshape(u_shape))
        tr = linalg.trace(eps)
        psi = 0.5 * LAM0 * tr**2 + MU0 * linalg.contract("...ij,...ji->...", eps, eps)
        return op.integrate(psi)

    return jax.jit(jax.grad(energy))


# The generic build only diverged from the closed form once the grid was fine
# enough to resolve near-null modes: Hex1R zeroed 1 mode at N=9 and N=15 but 9 at
# N=21.  These sizes are chosen so the comparison can actually fail.
AGREEMENT_N = {2: 21, 3: 21}


@pytest.mark.parametrize(("cls", "dim"), ISO_SCHEMES, ids=ISO_IDS)
def test_isotropic_matches_the_generic_preconditioner(cls, dim):
    """The closed form and the impulse-response build agree.

    This is the load-bearing test: the two arrive at the same operator by
    entirely different routes -- Woodbury on an identity-plus-rank-two symbol
    versus ``d`` impulse responses followed by an explicit ``d x d`` inverse.

    Two things it has to get right to be able to fail, both of which it
    originally got wrong:

    * the probe must be a **generic** vector, not ``K(v)``.  A residual in the
      range of ``K`` has no component on ``K``'s null modes, so both operators
      return zero there and any disagreement about *which* modes are null is
      invisible.
    * the grid must be fine enough for near-null modes to exist -- see
      ``AGREEMENT_N``.
    """
    n = AGREEMENT_N[dim]
    space = SpectralSpace(
        lengths=(1.0,) * dim, shape=(n,) * dim, transform=FFTTransform(dim=dim)
    )
    op = SpectralOperator(scheme=cls(space=space), space=space)
    K = _isotropic_reference(op, space, dim)
    V = _solution_space(space, dim)

    M_iso = make_isotropic_preconditioner(op.scheme, LAM0, MU0, V)
    M_gen = make_generic_preconditioner(K, V)

    r = jax.random.normal(jax.random.PRNGKey(0), (n**dim * dim,))
    got, ref = M_gen(r), M_iso(r)
    np.testing.assert_allclose(got, ref, atol=1e-10 * float(np.max(np.abs(ref))))


@pytest.mark.parametrize(("cls", "dim"), ISO_SCHEMES, ids=ISO_IDS)
def test_null_modes_are_gated_on_magnitude_not_determinant(cls, dim):
    """Both constructions must call the *same* modes null.

    ``det ~ lambda**d``, so gating the generic build on ``|det|`` made
    ``rtol=1e-12`` behave like ``1e-4`` and zero 9 modes where the closed form
    zeroed 1.  Comparing the counts directly catches that even if the surviving
    modes happen to agree.
    """
    n = AGREEMENT_N[dim]
    space = SpectralSpace(
        lengths=(1.0,) * dim, shape=(n,) * dim, transform=FFTTransform(dim=dim)
    )
    op = SpectralOperator(scheme=cls(space=space), space=space)
    V = _solution_space(space, dim)
    M_gen = make_generic_preconditioner(
        _isotropic_reference(op, space, dim), V
    )

    zeroed = int(jnp.sum(jnp.all(jnp.abs(M_gen.G_hat) == 0.0, axis=(-2, -1))))
    assert zeroed == 1, f"expected only the xi=0 mode to be gated, got {zeroed}"


@pytest.mark.parametrize(("cls", "dim"), ISO_SCHEMES, ids=ISO_IDS)
def test_isotropic_is_a_left_inverse_on_the_range(cls, dim):
    """``M^-1 K`` is the identity on the range: ``M K M == M``.

    Stated this way rather than ``M K v == v`` because the reference operator is
    genuinely singular -- rigid translation, plus the Nyquist corner for a
    reduced-integration stencil -- so a generic ``v`` is not recovered.
    """
    op, space = _operator(cls, dim)
    K = _isotropic_reference(op, space, dim)
    V = _solution_space(space, dim)
    M_inv = make_isotropic_preconditioner(op.scheme, LAM0, MU0, V)

    r = jax.random.normal(jax.random.PRNGKey(1), (N**dim * dim,))
    z = M_inv(r)
    np.testing.assert_allclose(M_inv(K(z)), z, atol=1e-10)


@pytest.mark.parametrize(("cls", "dim"), ISO_SCHEMES, ids=ISO_IDS)
def test_isotropic_projects_out_rigid_translation(cls, dim):
    """The ``xi = 0`` mode is annihilated, so the output has zero mean per component."""
    op, space = _operator(cls, dim)
    V = _solution_space(space, dim)
    M_inv = make_isotropic_preconditioner(op.scheme, LAM0, MU0, V)

    r = jax.random.normal(jax.random.PRNGKey(2), (N**dim * dim,))
    z = M_inv(r).reshape(space.shape + (dim,))
    np.testing.assert_allclose(jnp.mean(z, axis=tuple(range(dim))), 0.0, atol=1e-12)
    assert jnp.all(jnp.isfinite(z)), "null mode produced a non-finite result"


@pytest.mark.parametrize(("cls", "dim"), ISO_SCHEMES, ids=ISO_IDS)
def test_isotropic_stores_less_than_the_generic(cls, dim):
    """Matrix free: ``O(d)`` reals per voxel rather than ``O(d^2)`` complex.

    The point of keeping both. If this ever inverts, the isotropic form has lost
    its reason to exist.
    """
    op, space = _operator(cls, dim)
    K = _isotropic_reference(op, space, dim)
    V = _solution_space(space, dim)

    def nbytes(tree):
        return sum(
            leaf.nbytes
            for leaf in jax.tree_util.tree_leaves(tree)
            if hasattr(leaf, "nbytes")
        )

    iso = nbytes(make_isotropic_preconditioner(op.scheme, LAM0, MU0, V))
    gen = nbytes(make_generic_preconditioner(K, V))
    assert iso < gen, f"isotropic stored {iso} B, generic {gen} B"


def test_isotropic_survives_a_jit_boundary():
    op, space = _operator(Quad1RScheme, 2)
    V = _solution_space(space, 2)
    M_inv = make_isotropic_preconditioner(op.scheme, LAM0, MU0, V)

    r = jax.random.normal(jax.random.PRNGKey(3), (N**2 * 2,))
    out = jax.jit(lambda m, x: m(x))(M_inv, r)
    np.testing.assert_allclose(out, M_inv(r), atol=1e-14)


@pytest.mark.parametrize(
    ("cls", "dim"),
    [(Quad1RScheme, 2), (Hex1RScheme, 3), (Tetra2Scheme, 3), (FourierScheme, 2)],
    ids=["Quad1R", "Hex1R", "Tetra2", "Fourier"],
)
def test_isotropic_symbol_is_the_exact_acoustic_tensor(cls, dim):
    """The closed form equals the true quadrature-averaged acoustic tensor.

    ``A = mean_r [ mu0 |a_r|^2 I + lam0 conj(a_r) a_r^T + mu0 a_r a_r^H ]``

    The closed form uses only support 0 and drops an antisymmetric
    ``i (lam0 - mu0) (g h^T - h g^T)``.  That term vanishes here -- ``g`` and
    ``h`` are parallel for the single-support schemes, and TETRA2's mirror
    relation cancels it across supports -- so using one support is exact rather
    than approximate.  ``lam0 != mu0`` below, or the dropped term would be zero
    by construction and the test vacuous.
    """
    space = SpectralSpace(
        lengths=(1.0,) * dim, shape=(N,) * dim, transform=FFTTransform(dim=dim)
    )
    D = cls(space=space).gradient_operator  # (*spatial, n_quads, dim)
    eye = jnp.eye(dim)

    norm_sq = jnp.sum(D * jnp.conj(D), axis=-1).real
    outer_c = jnp.conj(D)[..., :, None] * D[..., None, :]
    outer_h = D[..., :, None] * jnp.conj(D)[..., None, :]
    exact = (
        MU0 * norm_sq[..., None, None] * eye + LAM0 * outer_c + MU0 * outer_h
    ).mean(axis=-3)

    a1 = D[..., 0, :]
    g, h = a1.real, a1.imag
    gg = jnp.sum(g * g, -1) + jnp.sum(h * h, -1)
    outer = g[..., :, None] * g[..., None, :] + h[..., :, None] * h[..., None, :]
    closed = MU0 * gg[..., None, None] * eye + (LAM0 + MU0) * outer

    scale = float(jnp.max(jnp.abs(exact)))
    np.testing.assert_allclose(closed, exact, atol=1e-12 * scale)


# ---------------------------------------------------------------------------
# adapt_green_preconditioner tests
# ---------------------------------------------------------------------------


def test_adapt_preconditioner_validation():
    space = SpectralSpace(
        lengths=(1.0, 1.0),
        shape=(8, 8),
        non_periodic_axes=("y",),
        transform=FFTTransform(dim=2),
    )
    V = SolutionSpace(space, u=Field(shape=(2,), y={0: "neumann", 1: "dirichlet"}))

    # Incompatible object: not a GreenPreconditioner
    with pytest.raises(TypeError, match="Expected a GreenPreconditioner"):
        adapt_green_preconditioner(lambda x: x, V)  # type: ignore[arg-type]

    # Incompatible grid: wrong extended_shape
    wrong_space = SpectralSpace(
        lengths=(1.0, 1.0), shape=(16, 16), transform=FFTTransform(dim=2)
    )
    V_wrong = SolutionSpace(wrong_space, u=Field(shape=(2,)))
    G_wrong = make_isotropic_preconditioner(
        QuadFullScheme(space=wrong_space), 1.0, 1.0, V_wrong
    )
    with pytest.raises(ValueError, match="does not match"):
        adapt_green_preconditioner(G_wrong, V)


def test_adapt_preconditioner_periodic_matches_green():
    space = SpectralSpace(
        lengths=(1.0, 1.0), shape=(9, 9), transform=FFTTransform(dim=2)
    )
    scheme = Quad1RScheme(space=space)
    V = SolutionSpace(space, u=Field(shape=(2,)))

    G = make_isotropic_preconditioner(scheme, 1.0, 2.0, V)
    M = G.adapt(V)

    key = jax.random.PRNGKey(123)
    r = jax.random.normal(key, V.shape)

    # Calling M with shaped array returns shaped array matching G(r.reshape(-1))
    z_m = M(r)
    z_g = G(r.reshape(-1)).reshape(V.shape)
    np.testing.assert_allclose(z_m, z_g, atol=1e-14)


def test_adapt_preconditioner_isotropic_is_left_inverse_on_range():
    """M K M == M on the physical domain with roller boundary conditions."""
    space = SpectralSpace(
        lengths=(1.0, 1.0),
        shape=(8, 8),
        non_periodic_axes=("y",),
        transform=FFTTransform(dim=2),
    )
    scheme = Quad1RScheme(space=space, allow_even_grid=True)
    op = SpectralOperator(scheme=scheme, space=space)

    lam0, mu0 = 1.5, 2.5
    V = SolutionSpace(space, u=Field(shape=(2,), y={0: "neumann", 1: "dirichlet"}))

    def K_ext_fn(u_ext):
        eps = op.sym_grad(u_ext)
        tr = linalg.trace(eps)
        sig = lam0 * tr[..., None, None] * jnp.eye(2) + 2.0 * mu0 * eps
        return -op.div(sig) * space.cell_volume

    def K_phys(u):
        u_ext = V.primal_to_extended(u).u
        force_ext = K_ext_fn(u_ext)
        return 0.5 * V.dual_to_physical(u=force_ext)

    M = make_isotropic_preconditioner(scheme, lam0, mu0, solution_space=V)

    key = jax.random.PRNGKey(42)
    r = V.apply_mask(jax.random.normal(key, V.shape))

    # Range test: M(K(M(r))) == M(r)
    z = M(r)
    MKz = M(K_phys(z))

    np.testing.assert_allclose(MKz, z, atol=1e-12)


def test_adapt_preconditioner_quadfull_roller_converges_in_one_step():
    """QuadFullScheme with roller BC converges to machine precision in 1 PCG step."""
    space = SpectralSpace(
        lengths=(1.0, 1.0),
        shape=(8, 8),
        non_periodic_axes=("y",),
        transform=FFTTransform(dim=2),
    )
    scheme = QuadFullScheme(space=space)
    op = SpectralOperator(scheme=scheme, space=space)

    lam0, mu0 = 1.5, 2.5
    V = SolutionSpace(space, u=Field(shape=(2,), y={0: "neumann", 1: "dirichlet"}))

    def energy_phys(u):
        u_ext = V.primal_to_extended(u).u
        eps = op.sym_grad(u_ext)
        tr = linalg.trace(eps)
        psi = 0.5 * lam0 * tr**2 + mu0 * linalg.contract("...ij,...ji->...", eps, eps)
        return op.integrate(psi)

    res_fn = jax.grad(energy_phys)

    def K_phys(v):
        return jax.jvp(res_fn, (V.zeros(),), (v,))[1]

    def energy_ext(u_ext_flat):
        u_ext = u_ext_flat.reshape(space.extended_shape + (2,))
        eps = op.sym_grad(u_ext)
        tr = linalg.trace(eps)
        psi = 0.5 * lam0 * tr**2 + mu0 * linalg.contract("...ij,...ji->...", eps, eps)
        return space.cell_volume * psi.mean(axis=-1).sum()

    M = make_generic_preconditioner(jax.grad(energy_ext), V)

    # Force vector in the range of K
    key = jax.random.PRNGKey(42)
    b = V.apply_mask(jax.random.normal(key, V.shape))
    b = b.at[:, :, 0].set(0.0)  # zero net lateral force
    b = V.apply_mask(b)

    # Wrap for 1D CG arithmetic
    def K_flat(v_flat):
        return K_phys(v_flat.reshape(V.shape)).reshape(-1)

    def M_flat(r_flat):
        return M(r_flat.reshape(V.shape)).reshape(-1)

    # Perform 1 iteration of PCG from u0 = 0
    r0 = b.reshape(-1)
    z0 = M_flat(r0)
    p0 = z0
    alpha0 = jnp.dot(r0, z0) / jnp.dot(p0, K_flat(p0))
    r1 = r0 - alpha0 * K_flat(p0)

    rel_res = float(jnp.linalg.norm(r1) / jnp.linalg.norm(r0))
    assert rel_res < 1e-12, f"Expected 1-step convergence, got relative residual {rel_res}"

