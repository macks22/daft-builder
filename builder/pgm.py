import itertools

import daft


def name_from_symbol(symbol):
    symbol = symbol.strip('$').replace('\\', '')
    if '{' in symbol:   # e.g. \tilde{X}
        start = symbol.index('{')
        end = symbol.index('}')
        return symbol[start + 1: end] + symbol[end + 1:] + '_' + symbol[:start]
    elif '^' in symbol:  # e.g. \sigma_c^2
        base, exp = symbol.split('^')
        if exp != '2':
            raise ValueError('unable to handle names with exponents not equal to 2')
        return base + '_' + 'sq'
    else:
        return symbol


def node_bounds(*nodes):
    """Get the min and max for the x- and y-coordinates for an iterable of `daft.Node`s."""
    xs = list(n.x for n in nodes)
    ys = list(n.y for n in nodes)
    return (min(xs), max(xs)), (min(ys), max(ys))


def plate_rect_shape(*nodes):
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


class _PGM(daft.PGM):
    """Patched daft PGM class with useful methods for adding batches of edges and nodes."""

    def add_nodes(self, nodes):
        for node in nodes:
            self.add_node(node)

    def add_plates(self, plates):
        for plate in plates:
            self.add_plate(plate)

    def add_edges(self, names):
        for from_name, to_name in names:
            self.add_edge(from_name, to_name)


class Node:
    """Helper class to build `daft.Node` objects with edges to other nodes."""

    def __init__(self, *args, **kwargs):
        name = name_from_symbol(args[0])
        args = list(args)
        args.insert(0, name)

        kwargs.setdefault('scale', 2)
        if 'fixed' in kwargs:
            kwargs['offset'] = (0, 10)

        self.args = args
        self.kwargs = kwargs
        self.edges_to = []

    def with_edges_to(self, *names):
        self.edges_to += names
        return self

    def build(self):
        return daft.Node(*self.args, **self.kwargs)


class Plate:
    """Helper class to build `daft.Plate` objects with nodes inside them."""

    def __init__(self, label, **kwargs):
        kwargs['label'] = label
        kwargs.setdefault('shift', -0.1)
        kwargs.setdefault('bbox', {"color": "none"})
        kwargs.setdefault('position', 'bottom right')

        self.kwargs = kwargs
        self.nodes = []
        self.built_nodes = []

    def with_nodes(self, *node_builders):
        self.nodes += node_builders
        return self

    def build(self):
        if not self.nodes:
            raise ValueError('plate must have at least one node')

        # Position plate so that it encompasses all its nodes
        # with room to spare for label and margins.
        # Build the nodes first so we can use the (x, y) coords from the objects.
        self.built_nodes = [node_builder.build() for node_builder in self.nodes]
        rect = plate_rect_shape(*self.built_nodes)
        return daft.Plate(rect, **self.kwargs)


class PGM:
    """Helper class to build PGMs from the bottom-up."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('origin', (0, 0))
        kwargs.setdefault('grid_unit', 4)
        kwargs.setdefault('label_params', {'fontsize': 18})
        kwargs.setdefault('observed_style', 'shaded')

        self.args = args
        self.kwargs = kwargs
        self.nodes = []
        self.plates = []

    def with_plate(self, plate_builder):
        self.plates.append(plate_builder)
        return self

    def with_nodes(self, *node_builders):
        self.nodes += node_builders
        return self

    def build(self):
        pgm = _PGM(*self.args, **self.kwargs)

        # Build plates and the nodes within them first.
        plates = [builder.build() for builder in self.plates]
        plate_nodes = itertools.chain.from_iterable(
            plate.built_nodes for plate in self.plates)

        # Next build standalone nodes.
        other_nodes = [builder.build() for builder in self.nodes]
        all_nodes = other_nodes + list(plate_nodes)

        # Now get all the edge pairs.
        edge_pairs = self.get_edge_pairs()

        # Finally, add all plates, nodes, and edges to PGM
        pgm.add_plates(plates)
        pgm.add_nodes(all_nodes)
        pgm.add_edges(edge_pairs)

        return pgm

    def get_edge_pairs(self):
        plate_node_builders = itertools.chain.from_iterable(
            plate.nodes for plate in self.plates)
        all_node_builders = self.nodes + list(plate_node_builders)

        for node_builder in all_node_builders:
            name = name_from_symbol(node_builder.args[0])
            for to_node_name in node_builder.edges_to:
                yield (name, to_node_name)
