import numpy as np
import pytest

from xpektra import make_field


@pytest.mark.parametrize(
    "dim, shape, rank, expected_shape",
    [
        (1, (8,), 0, (8,)),
        (1, (8,), 1, (8, 1)),
        (1, (8,), 2, (8, 1, 1)),
        (2, (4, 4), 0, (4, 4)),
        (2, (4, 4), 1, (4, 4, 2)),
        (2, (4, 4), 2, (4, 4, 2, 2)),
        (3, (4, 4, 4), 0, (4, 4, 4)),
        (3, (4, 4, 4), 1, (4, 4, 4, 3)),
        (3, (4, 4, 4), 2, (4, 4, 4, 3, 3)),
    ],
)
def test_make_field_shape(dim, shape, rank, expected_shape):
    f = make_field(dim=dim, shape=shape, rank=rank)
    assert f.shape == expected_shape


def test_make_field_zeros():
    f = make_field(dim=2, shape=(4, 4), rank=2)
    assert np.all(f == 0)


def test_make_field_default_dtype():
    f = make_field(dim=2, shape=(4, 4), rank=0)
    assert np.issubdtype(f.dtype, np.floating)


def test_make_field_complex_dtype():
    f = make_field(dim=2, shape=(4, 4), rank=0, dtype=complex)
    assert np.issubdtype(f.dtype, np.complexfloating)


def test_make_field_fill_value_ones():
    f = make_field(dim=2, shape=(4, 4), rank=2, fill_value=1.0)
    assert np.all(f == 1.0)


def test_make_field_fill_value_negative():
    f = make_field(dim=2, shape=(4, 4), rank=1, fill_value=-3.5)
    assert np.all(f == -3.5)


def test_make_field_fill_value_zero_explicit():
    f = make_field(dim=2, shape=(4, 4), rank=0, fill_value=0)
    assert np.all(f == 0)
