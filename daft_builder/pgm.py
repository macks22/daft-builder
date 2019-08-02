"""
Core builder interfaces.

"""
import logging
import itertools
import copy

import daft
from toposort import toposort

from daft_builder import utils

DEFAULT_VERTICAL_OFFSET = 1
DEFAULT_HORIZONTAL_OFFSET = 0.8

logger = logging.getLogger(__name__)


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
            logger.debug(f"adding edge from {from_name} to {to_name}")
            self.add_edge(from_name, to_name)


class Node:
    """Helper class to build `daft.Node` objects with edges to other nodes."""

    _placement_kwargs = ('xy',
                         'above', 'above_l', 'above_r',
                         'below', 'below_l', 'below_r',
                         'left_of', 'left_of_a', 'left_of_b',
                         'right_of', 'right_of_a', 'right_of_b')

    def __init__(self, symbol, **kwargs):
        """
        Placement can be accomplished with at most one of:
            xy, above, above_l, above_r, below, below_l, below_r,
            left_of, left_of_a, left_of_b, right_of, right_of_a, right_of_b

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
        self.symbol = symbol
        self.name = self.kwargs['name']
        self.edges_to = []

        self.plate = None

    def __repr__(self):
        return f'{self.__class__.__name__}({self.symbol}, **{self.kwargs})'

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if other is None or not hasattr(other, 'name'):
            return False

        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

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
            name = utils.name_from_symbol(symbol)
            kwargs['name'] = name

        kwargs.setdefault('scale', 2)
        if 'fixed' in kwargs and 'offset' not in kwargs:
            kwargs['offset'] = (0, -25) if self.placement.startswith('below') else (0, 10)

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
            if self.anchor_node is None:
                raise ValueError(f"Param {self.name} must specify node it's a parameter of via "
                                 f"the `of` kwarg or a relative placement kwarg")
            self.edges_to += [self.anchor_node]
        elif isinstance(of, (str, int, float)):
            self.edges_to += [of]
        elif hasattr(of, '__iter__'):
            self.edges_to += list(of)
        else:
            raise ValueError(f"unrecognized type for argument 'of': {of} ({type(of)})")


class HyperParam(Param):
    def __init__(self, symbol, of=None, **kwargs):
        kwargs.setdefault('fixed', True)
        super().__init__(symbol, of, **kwargs)


class Text(Node):
    """Convenience class for a node being used to place text in a PGM."""
    def __init__(self, text, name=None, **kwargs):
        kwargs['plot_params'] = {**kwargs.get('plot_params', {}), **{'ec': 'none'}}
        super().__init__(text, name=name, **kwargs)


class Plate:
    """Helper class to build `daft.Plate` objects with nodes inside them."""

    def __init__(self, label, **kwargs):
        self.label = label
        kwargs['label'] = self.label

        kwargs.setdefault('shift', -0.1)
        kwargs.setdefault('bbox', {"color": "none"})
        kwargs.setdefault('position', 'bottom right')

        self.kwargs = kwargs
        self.nodes = []

        self.x = None
        self.y = None
        self.width = None
        self.height = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.kwargs['label']}, **{self.kwargs})"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return self.kwargs['label'] == other.kwargs['label']

    @property
    def rect(self):
        if (self.x is None or
                self.y is None or
                self.width is None or
                self.height is None):
            return None

        return self.x, self.y, self.width, self.height

    @rect.setter
    def rect(self, rect):
        self.x, self.y, self.width, self.height = rect

    def place(self):
        self.rect = utils.bound_nodes_with_rect(*self.nodes)

    def deconflict_placement(self, other):
        logger.debug('Detected overlapping plates: %s, %s', self.label, other.label)
        if self.same_nodes_as(other):  # complete overlap
            logger.debug(f'Plate {self.label} has same nodes as other {other.label}')
            other.surround(self)
        elif self.contains_nodes_of(other):  # superset
            logger.debug(f'Plate {self.label} contains nodes of other {other.label}')
            self.surround(other)
        elif other.contains_nodes_of(self):  # subset
            logger.debug(f'Plate {other.label} contains nodes of other {self.label}')
            other.surround(self)
        else:  # both plates have nodes the other doesn't
            logger.debug(f"Both plates have nodes the other doesn't: "
                         f"{self.label}, {other.label}")
            # shift other out of the way
            other.y -= 0.2
            other.height += 0.2
            self.height += 0.1

    def surround(self, other, amount=0.15):
        x = y = width = height = 0  # offsets

        shares_left = round(self.x, 2) == round(other.x, 2)
        if shares_left:
            logger.debug('shares left')
            x = -amount
            width = amount

        shares_right = round(self.x + self.width, 2) == round(other.x + other.width, 2)
        if shares_right:
            logger.debug('shares right')
            y = -amount
            width += amount
            height = 2 * amount

        shares_bottom = round(self.y, 2) == round(other.y, 2)
        if shares_bottom:
            logger.debug('shares bottom')
            y = -amount
            height = amount

        shares_top = round(self.y + self.height, 2) == round(other.y + other.height, 2)
        if shares_top:
            logger.debug('shares top')
            height += amount

        self.rect = (self.x + x, self.y + y,
                     self.width + width, self.height + height)

    def shares_nodes_with(self, other):
        if not hasattr(other, 'nodes'):
            raise ValueError("can't compare nodes to object without 'nodes' attribute.")

        return bool(set(self.nodes).intersection(set(other.nodes)))

    def same_nodes_as(self, other):
        return set(self.nodes) == set(other.nodes)

    def contains_nodes_of(self, other):
        if not hasattr(other, 'nodes'):
            raise ValueError("can't compare nodes to object without 'nodes' attribute.")

        return set(self.nodes).issuperset(set(other.nodes))

    def with_nodes(self, *node_builders):
        for node in node_builders:
            # These can either be actual Node daft_builder objects or names of
            # nodes already added to the PGM elsewhere (or to be added later)
            if not isinstance(node, str):
                node.plate = self

        self.nodes += node_builders
        return self

    def build(self):
        if not self.nodes:
            raise ValueError('plate must have at least one node')

        if self.rect is None:
            self.place()

        return daft.Plate(self.rect, **self.kwargs)


class PGM:
    """Helper class to build PGMs from the bottom-up."""

    def __init__(self, *, vertical_offset=None, horizontal_offset=None, **kwargs):
        kwargs.setdefault('origin', (0, 0))
        kwargs.setdefault('grid_unit', 4)
        kwargs.setdefault('label_params', {'fontsize': 18})
        kwargs.setdefault('observed_style', 'shaded')

        self.kwargs = kwargs
        self.nodes = []
        self.plates = []

        self.vertical_offset = \
            DEFAULT_VERTICAL_OFFSET if vertical_offset is None else vertical_offset
        self.horizontal_offset = \
            DEFAULT_HORIZONTAL_OFFSET if horizontal_offset is None else horizontal_offset

    def with_plate(self, plate_builder):
        self.plates.append(plate_builder)
        return self

    def with_nodes(self, *node_builders):
        self.nodes += node_builders
        return self

    def get_node(self, name):
        for node in self.nodes:
            if node.name == name:
                return node

        raise KeyError(f"No node with name {name}")

    @property
    def plate_node_builders(self):
        return itertools.chain.from_iterable(
            plate.nodes for plate in self.plates)

    @property
    def all_node_builders(self):
        return self.nodes + list(self.plate_node_builders)

    def build(self):
        # First fill in any node builders that are referenced by name in the plates
        self.fill_nodes_refd_by_name()

        # Handle placement of all nodes before building anything.
        self.place_nodes()

        # Next place all plates
        self.place_plates()

        if 'shape' not in self.kwargs:
            self.kwargs['shape'] = self.place()

        pgm = _PGM(**self.kwargs)

        # Build plates and nodes.
        plates = [builder.build() for builder in self.plates]
        nodes = [builder.build() for builder in self.all_node_builders]

        # Finally, add all plates, nodes, and edges to PGM
        pgm.add_plates(plates)
        pgm.add_nodes(nodes)
        pgm.add_edges(self.get_edge_pairs())

        return pgm

    def copy(self):
        return copy.deepcopy(self)

    def fill_nodes_refd_by_name(self):
        """Sub in actual node daft_builder anywhere node was referenced by name in any plates."""
        node_mapping = {n.name: n for n in self.all_node_builders if not isinstance(n, str)}
        for plate in self.plates:
            plate.nodes = [node_mapping[n] if isinstance(n, str) else n for n in plate.nodes]

    def place_nodes(self):
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
            x, y = (anchor_x, anchor_y + self.vertical_offset)
        elif builder.placement.startswith('below'):
            x, y = (anchor_x, anchor_y - self.vertical_offset)
        elif builder.placement.startswith('left_of'):
            x, y = (anchor_x - self.horizontal_offset, anchor_y)
            # Adjust for plate around anchor, if present
            if anchor.plate and not anchor.in_same_plate(builder):
                x -= 0.1
        else:  # right_of
            x, y = (anchor_x + self.horizontal_offset, anchor_y)
            # Adjust for plate around anchor, if present
            if anchor.plate and not anchor.in_same_plate(builder):
                x += 0.1

        # Do shifting if specified
        if builder.placement.endswith('_l'):
            x -= 0.3
        elif builder.placement.endswith('_r'):
            x += 0.3
        elif builder.placement.endswith('_a'):
            y += 0.3
        elif builder.placement.endswith('_b'):
            y -= 0.3

        return x + builder.shift_x, y + builder.shift_y

    def place_plates(self):
        for plate in self.plates:
            plate.place()

        # Need to reposition any overlapping plates.
        self.deconflict_plate_placement()

    def deconflict_plate_placement(self):
        # We'll figure this out by looking at the nodes inside them.
        # We could look at the corners, but some plates may be exactly in the middle of others,
        # and the corners method would fail to find those.
        all_plates = self.plates[::-1]
        while all_plates:
            plate = all_plates.pop()
            for other in all_plates:
                if plate.shares_nodes_with(other):
                    plate.deconflict_placement(other)

    def place(self):
        x, y, width, height = utils.bound_nodes_with_rect(*self.all_node_builders)
        x_units = x + width
        y_units = y + height

        # Need to expand further to accommodate plates, if present
        # Right now, we just assume plates are present all around.
        # TODO: actually figure out if plates are present on the outer bounds and only expand if so.
        if self.plates:
            x_units += 0.3
            y_units += 0.3

        return x_units, y_units

    def get_edge_pairs(self):
        for node_builder in self.all_node_builders:
            for to_node_name in node_builder.edges_to:
                yield (node_builder.name, to_node_name)

