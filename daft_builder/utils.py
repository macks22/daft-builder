"""
Utility functions.

"""


def name_from_symbol(symbol):
    starts_with_modifier = symbol.startswith('$\\')
    symbol = symbol.strip('$').replace('\\', '')

    if starts_with_modifier and '{' in symbol:   # e.g. \tilde{X}
        start = symbol.index('{')
        end = symbol.index('}')
        without_modifier = symbol[start + 1: end] + symbol[end + 1:]
        modifier = symbol[:start]
        base, extras = without_modifier.split('_', 1)
        symbol = '_'.join([base, modifier, extras])

    if '_{' in symbol:  # e.g. X_{i, j}
        without_sub, rest = symbol.split('_{', 1)
        subscript, rest = rest.split('}', 1)
        subscript = subscript.replace(',', '').replace(' ', '')
        symbol = '_'.join([without_sub, subscript]) + rest

    if '^' in symbol:  # e.g. \sigma_c^2
        base, exp = symbol.split('^')
        if exp != '2':
            raise ValueError('unable to handle names with exponents not equal to 2')
        symbol = base + '_' + 'sq'

    return symbol


def node_bounds(*nodes):
    """Get the min and max for the x- and y-coordinates for an iterable of `daft.Node`s."""
    xs = list(n.x for n in nodes)
    ys = list(n.y for n in nodes)
    return (min(xs), max(xs)), (min(ys), max(ys))


def bound_nodes_with_rect(*nodes):
    """Figure out plate rectangle shape arguments that will result in a plate
    that fully encompasses all the given nodes with enough space to spare for
    a label and margins.

    """
    (min_x, max_x), (min_y, max_y) = node_bounds(*nodes)
    plate_x = min_x - 0.4
    plate_y = min_y - 0.35

    plate_x_right = max_x + 0.8
    plate_y_top = max_y + 0.75

    plate_width = round(plate_x_right - min_x, 2)
    plate_height = round(plate_y_top - min_y, 2)

    return plate_x, plate_y, plate_width, plate_height
