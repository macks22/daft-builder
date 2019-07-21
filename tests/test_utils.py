import matplotlib; matplotlib.use('Agg')
import pytest

from daft_builder import utils


@pytest.mark.parametrize('symbol,expected', [
    ("test", "test"),
    (r"$X$", "X"),
    (r"$\theta$", "theta"),
    (r"$\Sigma$", "Sigma"),
    (r"$\sigma_c", "sigma_c"),
    (r"$\sigma_c^2", "sigma_c_sq"),
    (r"$X_{i, j}", "X_ij"),
    (r"$\tilde{X}_i", "X_tilde_i"),
    (r"$\tilde{X}_{i,j}", "X_tilde_ij"),
    (r"$\tilde{X}_{i,j}^2", "X_tilde_ij_sq")
])
def test_name_from_symbol(symbol, expected):
    assert utils.name_from_symbol(symbol) == expected
