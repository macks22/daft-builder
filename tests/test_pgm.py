import matplotlib; matplotlib.use('Agg')
from daft_builder import pgm

import pytest


def test_Param_init():
    param = pgm.Param(r"$y$", xy=(0.5, 0.5), of=["x"])
    assert param.name == "y"
    assert param.x, param.y == (0.5, 0.5)
    assert param.anchor_node is None
    assert param.edges_to == ["x"]


@pytest.mark.parametrize("of", [
    22, 22.1, -1, 0
])
def test_Param_init_referring_to_number_named_nodes(of):
    param = pgm.Param(r"$y$", xy=(1, 1), of=of)
    assert param.edges_to == [of]


def test_init_Param_of_multiple_nodes():
    param = pgm.Param(r"$y$", xy=(1, 1), of=["x", "w"])
    assert param.edges_to == ["x", "w"]


def test_Param_init_requires_valid_anchor():
    with pytest.raises(ValueError):
        pgm.Param(r"$y$", xy=(1, 1))


def test_Text_init():
    t = pgm.Text("some text", xy=(1, 1))
    assert t.name == "some text"
    assert t.kwargs['plot_params'] == {"ec": "none"}

    t = pgm.Text("some text", "t", xy=(1, 1))
    assert t.name == "t"
    assert t.kwargs['plot_params'] == {"ec": "none"}
