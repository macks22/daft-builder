import itertools
from functools import reduce

import daft


def name_from_symbol(symbol):
    symbol = symbol.strip('$').replace('\\', '')
    if '{' in symbol:   # e.g. \tilde{X}
        start = symbol.index('{')
        end = symbol.index('}')
        without_modifier = symbol[start + 1: end] + symbol[end + 1:]
        modifier = symbol[:start]
        base, extras = without_modifier.split('_', 1)
        return '_'.join([base, modifier, extras])
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


def toposort(dep_graph):
    """Perform a topological sort of a dependency graph, returning the
    names of nodes in topologically ordered sets.
    Dependencies are expressed as a dictionary whose keys are items
    and whose values are a set of dependent items. Output is a list of
    sets in topological order. The first set consists of items with no
    dependencies. Each subsequent set consists of items that depend upon
    items in the preceding sets.

    Args:
        dep_graph (dict): directed graph represented as adjacency list,
            with string keys and sets of strings as values. Any nodes
            without dependencies can either be included as keys with an
            empty set for values, or excluded, which will be interpreted
            in the same manner.

    Returns:
        generator[set]: ordered sets of nodes, such that the nodes in each
            set do not depend on any of the nodes in subsequent sets, or
            on each other.

    Examples:
        A graph with only 1-hop dependencies:
        >>> graph = {"a": {"b", "c", "d"}, "c": {"d"}}
        >>> tuple(toposort(graph))  # ({'b', 'd'}, {'c'}, {'a'})
        And here is another with 2-hop dependencies:
        >>> graph = {"a": {"b", "c", "d"}, "c": {"d"}, "e": {"g", "f", "q"}}
        >>> tuple(toposort(graph))  # ({'b', 'd', 'g', 'f', 'q'}, {'c', 'e'}, {'a'})

    Raises:
        ValueError: if the dependency graph contains a cycle.
    """
    # Special case empty input.
    if not dep_graph:
        return

    # Copy the input so as to leave it unmodified.
    dep_graph = {k: set(v) for k, v in dep_graph.items()}

    # Automatically add an empty set for any dependencies not included as keys.
    all_deps = itertools.chain.from_iterable(v for v in dep_graph.values())
    deps_without_deps = [dep for dep in all_deps if dep not in dep_graph]
    dep_graph.update({dep: set() for dep in deps_without_deps})

    # Ignore self dependencies.
    for k, v in dep_graph.items():
        v.discard(k)

    # Find all items that don't depend on anything.
    extra_items_in_deps = reduce(set.union, dep_graph.values()) - set(dep_graph.keys())

    # Add empty dependencies where needed.
    dep_graph.update({item: set() for item in extra_items_in_deps})

    while True:
        ordered = set(item for item, dep in dep_graph.items() if len(dep) == 0)
        if not ordered:
            break

        yield ordered
        dep_graph = {item: (dep - ordered)
                     for item, dep in dep_graph.items()
                     if item not in ordered}

    if dep_graph:
        raise ValueError(f'detected circular dependency in graph: {dep_graph}')


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

    _placement_kwargs = ('xy',
                         'above', 'above_l', 'above_r',
                         'below', 'below_l', 'below_r',
                         'left_of', 'right_of')

    def __init__(self, symbol, **kwargs):
        """
        Placement can be accomplished with at most one of:
            xy, above, above_l, above_r, below, below_l, below_r, left_of, right_of

        Args:
            symbol (str): symbol to label node with. Can be LaTeX expression.
            **kwargs: passed through directly to `daft.Node` constructor. Some of these
                are special and are involved the placement of the node. The position can
                be specified explicitly using:
                    xy (tuple): x- and y-coordinates.
                Or an anchor node and this node's position relative to it can be specified
                by one of:
                    above (str): name of node to place this node above. Can also use
                        above_l and above_r for above with a left or right offset.
                    below (str): name of node to place this node below. Can also use
                        below_l and below_r for below with a left or right offset.
                    left_of (str): name of node to place this node to the left of.
                    right_of (str): name of node to place this node to the right of.
                Finally, a shift from the relative position can be specified by:
                    shift_x (float): shift in x coords
                    shift_y (float): shift in y coords
                These last two parameters are ignored if `xy` is passed.

        Raises:
            ValueError: if more than one of the placement kwargs is given.
        """
        self.x = None
        self.y = None
        self.placement = None
        self.anchor_node = None
        self.shift_x = kwargs.pop('shift_x', 0)
        self.shift_y = kwargs.pop('shift_y', 0)
        self.set_placement(kwargs)

        self.kwargs = self.add_kwarg_defaults(symbol, kwargs)
        self.sybmol = symbol
        self.name = self.kwargs['name']
        self.edges_to = []

        self.plate = None

    def in_same_plate(self, other):
        """Compare containing plates of Node builders.

        Args:
            other (Node): other node to compare plate to.

        Returns:
            True if the nodes are in the same plate, else False.
        """
        if self.plate is None:
            return False
        if other.plate is None:
            return False

        return self.plate == other.plate

    def set_placement(self, kwargs):
        placement_kwargs = {name: kwargs.pop(name) for name in self._placement_kwargs
                            if name in kwargs}
        num_placements_given = len(placement_kwargs)
        if num_placements_given != 1:
            raise ValueError(f'{self.__class__.__name__}__init__ can handle at most one of the '
                             f'placement kwargs ({self._placement_kwargs}) '
                             f'but was given {num_placements_given}')

        if 'xy' in placement_kwargs:
            self.x, self.y = placement_kwargs['xy']
        else:
            self.placement, self.anchor_node = next(iter(placement_kwargs.items()))

    def add_kwarg_defaults(self, symbol, kwargs):
        kwargs = kwargs.copy()

        kwargs['content'] = symbol
        name = kwargs.get('name')
        if name is None:
            name = name_from_symbol(symbol)
            kwargs['name'] = name

        kwargs.setdefault('scale', 2)
        if 'fixed' in kwargs:
            kwargs['offset'] = (0, 10)

        return kwargs

    def with_edges_to(self, *names):
        self.edges_to += names
        return self

    def build(self):
        kwargs = self.kwargs.copy()
        kwargs['x'], kwargs['y'] = self.x, self.y
        return daft.Node(**kwargs)


class Data(Node):
    def __init__(self, symbol, **kwargs):
        kwargs.setdefault('observed', True)
        super().__init__(symbol, **kwargs)


class Param(Node):
    def __init__(self, symbol, of=None, **kwargs):
        super().__init__(symbol, **kwargs)
        if of is None:
            self.edges_to += [self.anchor_node]
        elif isinstance(of, str):
            self.edges_to += [of]
        elif hasattr(of, '__iter__'):
            self.edges_to += list(of)
        else:
            raise ValueError(f"unrecognized type for argument 'of': {of} ({type(of)})")


class HyperParam(Param):
    def __init__(self, symbol, of=None, **kwargs):
        kwargs.setdefault('fixed', True)
        super().__init__(symbol, of, **kwargs)


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

    def __eq__(self, other):
        return self.kwargs['label'] == other.kwargs['label']

    def with_nodes(self, *node_builders):
        for node in node_builders:
            node.plate = self

        self.nodes += node_builders
        return self

    def build_nodes(self):
        self.built_nodes = [node_builder.build() for node_builder in self.nodes]

    def build(self):
        if not self.nodes:
            raise ValueError('plate must have at least one node')

        # Position plate so that it encompasses all its nodes
        # with room to spare for label and margins.
        # Build the nodes first so we can use the (x, y) coords from the objects.
        if not self.built_nodes:
            self.build_nodes()
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

    @property
    def plate_node_builders(self):
        return itertools.chain.from_iterable(
            plate.nodes for plate in self.plates)

    @property
    def all_node_builders(self):
        return self.nodes + list(self.plate_node_builders)

    def build(self):
        pgm = _PGM(*self.args, **self.kwargs)

        # Handle placement of all nodes before building anything.
        self.place_all_nodes()

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

    def place_all_nodes(self):
        # Build dep graph
        builders = self.all_node_builders
        dep_graph = {n.name: set() if n.anchor_node is None else {n.anchor_node}
                     for n in builders}
        name_batches = toposort(dep_graph)
        builder_map = {n.name: n for n in builders}

        # Place nodes in each batch
        for names in name_batches:
            for name in names:
                builder = builder_map[name]
                if builder.anchor_node is not None:
                    anchor = builder_map[builder.anchor_node]
                    builder.x, builder.y = self.place_node(builder, anchor)

    def place_node(self, builder, anchor):
        """Get x, y coords for a particular node relative to its anchor node.

        Args:
            builder (Node): node to place.
            anchor (Node): node this one is anchored to.

        Returns:
             tuple: x, y coords for the node's placement.
        """
        anchor_x, anchor_y = anchor.x, anchor.y

        if builder.placement.startswith('above'):
            x, y = (anchor_x, anchor_y + 1)
        elif builder.placement.startswith('below'):
            x, y = (anchor_x, anchor_y - 1)
        elif builder.placement == 'left_of':
            x, y = (anchor_x - 0.6, anchor_y)
            # Adjust for plate around anchor, if present
            if anchor.plate and not anchor.in_same_plate(builder):
                x -= 0.3
        else:  # right_of
            x, y = (anchor_x + 0.6, anchor_y)
            # Adjust for plate around anchor, if present
            if anchor.plate and not anchor.in_same_plate(builder):
                x += 0.3

        # Do shifting if specified
        if builder.placement.endswith('_l'):
            x -= 0.3
        elif builder.placement.endswith('_r'):
            x += 0.3

        return x + builder.shift_x, y + builder.shift_y

    def get_edge_pairs(self):
        for node_builder in self.all_node_builders:
            for to_node_name in node_builder.edges_to:
                yield (node_builder.name, to_node_name)

