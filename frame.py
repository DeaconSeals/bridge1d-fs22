import matplotlib.collections
import matplotlib.pyplot as pyplot
import numpy
import numpy.linalg
import scipy.linalg

from typing import ClassVar, TypedDict, Optional, List, Dict, Tuple

import itertools
import math
import random

# Reference: https://www.gitmechanics.com/CEE321/Direct-stiffness/Main.pdf


PointList = List[Tuple[float, float]]

# Fun fact: the gravitational acceleration on the surface of the earth varies between 9.76 and 9.83 m/s^2 depending on latitude and altitude!
GRAVITATIONAL_ACCELERATION = 9.8


class Node:
    x: float
    y: float
    fixed_x: bool
    fixed_y: bool
    fixed_angle: bool

    def __init__(self, x: float, y: float, fixed_x: bool = False, fixed_y: Optional[bool] = None,
                 fixed_angle: Optional[bool] = None):
        self.x = x
        self.y = y
        self.fixed_x = fixed_x
        if fixed_y is not None:
            self.fixed_y = fixed_y
        else:
            self.fixed_y = fixed_x
        if fixed_angle is not None:
            self.fixed_angle = fixed_angle
        else:
            self.fixed_angle = self.fixed_y

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return self.__str__()


class Element:
    start: Node
    end: Node
    elasticity: float  # Young's modulus (E)
    area: float  # Cross-sectional area (A)
    moment: float  # Second moment of area / area moment of inertia (I)
    radius: float  # Centroidal distance of cross-section (might not be half-width if asymmetric)
    density: float
    yield_strength: float
    compressive_strength: float
    flexural_strength: float

    vector: numpy.ndarray
    angle: float
    length: float
    unit_vector: numpy.ndarray
    weight: float
    transformation_matrix: numpy.ndarray
    stiffness_matrix: numpy.ndarray

    # S355 steel, 400mm, 20mm thick square hollow sections as default
    default_elasticity: ClassVar[float] = 210 * 1000 ** 3  # 210 GPa
    default_area: ClassVar[float] = 0.0304  # m^2 (400mm 20mm thick square)
    default_moment: ClassVar[float] = 0.734 * 1000 ** -1  # m^4 (400mm 20mm thick square)
    default_radius: ClassVar[float] = 0.400 / 2  # m (symmetric section)
    default_density: ClassVar[float] = 7850  # kg/m^3
    default_yield_strength: ClassVar[float] = 355 * 1000 ** 2  # 355 MPa
    default_compressive_strength: ClassVar[float] = 355 * 1000 ** 2  # 355 MPa
    default_flexural_strength: ClassVar[float] = 355 * 1000 ** 2  # 355 MPa

    def __init__(self, start: Node, end: Node, elasticity: float = None, area: float = None, moment: float = None,
                 radius: float = None, density: float = None, yield_strength: float = None,
                 compressive_strength: float = None, flexural_strength: float = None):
        self.start = start
        self.end = end
        self.elasticity = elasticity if elasticity is not None else Element.default_elasticity
        self.area = area if area is not None else Element.default_area
        self.moment = moment if moment is not None else Element.default_moment
        self.radius = radius if radius is not None else Element.default_radius
        self.density = density if density is not None else Element.default_density
        self.yield_strength = yield_strength if yield_strength is not None else Element.default_yield_strength
        self.compressive_strength = compressive_strength if compressive_strength is not None else Element.default_compressive_strength
        self.flexural_strength = flexural_strength if flexural_strength is not None else Element.default_flexural_strength

        self.calculate_derived_values()

    def calculate_derived_values(self):
        self.vector = numpy.array([self.end.x - self.start.x, self.end.y - self.start.y])
        self.angle = math.atan2(self.vector[1], self.vector[0])
        self.length = numpy.linalg.norm(self.vector)
        self.unit_vector = self.vector / self.length
        self.weight = self.length * self.area * self.density

        e = self.area * self.elasticity / self.length  # AE/L
        d = 2 * self.elasticity * self.moment / self.length  # 2EI/L
        c = 2 * d  # 4EI/L
        b = 3 * d / self.length  # 6EI/L^2
        a = 2 * b / self.length  # 12EI/L^3
        local_stiffness_matrix = numpy.array((
            (e, 0, 0, -e, 0, 0),
            (0, a, b, 0, -a, b),
            (0, b, c, 0, -b, d),
            (-e, 0, 0, e, 0, 0),
            (0, -a, -b, 0, a, -b),
            (0, b, d, 0, -b, c)
        ))

        small_transformation_matrix = numpy.array((
            (self.unit_vector[0], -self.unit_vector[1], 0),
            (self.unit_vector[1], self.unit_vector[0], 0),
            (0, 0, 1)
        ))
        zero_transformation_matrix = numpy.zeros((3, 3))
        self.transformation_matrix = numpy.block([
            [small_transformation_matrix, zero_transformation_matrix],
            [zero_transformation_matrix, small_transformation_matrix]
        ])

        self.stiffness_matrix = self.transformation_matrix @ local_stiffness_matrix @ self.transformation_matrix.transpose()

    def get_failure_ratio(self, displacement_vector: numpy.ndarray) -> float:
        # If return value > 1, tensile failure; if return value < 1, compressive failure
        local_displacement_vector = self.transformation_matrix.transpose() @ displacement_vector
        local_force_vector = self.transformation_matrix.transpose() @ self.stiffness_matrix @ displacement_vector
        axial_stress = (local_displacement_vector[3] - local_displacement_vector[0]) * \
                       self.elasticity * self.area / self.length

        # Assumes load is only at nodes, distributed load causes worse bending mid-element.
        # For symmetrical cross-sections this gives equal compressive and tensile loads.
        maximum_bending_moment = max(abs(local_force_vector[2]), abs(local_force_vector[5]))
        maximum_bending_stress = maximum_bending_moment * self.radius / self.moment

        if axial_stress >= 0:
            # Tension, check for rupture
            # Inaccurate for asymmetric cross-sections!
            return (axial_stress + maximum_bending_stress) / self.get_yield_force()
        else:
            # Compression, check for buckling
            # https://s3.amazonaws.com/suncam/docs/307.pdf
            axial_ratio = -axial_stress / self.get_compressive_failure_force()
            # Ignoring amplification, presumably that's taken care of in displacement?
            moment_ratio = maximum_bending_moment / self.get_ultimate_moment()
            return -(axial_ratio + moment_ratio)  # Buckling occurs if the sum of ratios is greater than one

    def get_yield_force(self) -> float:
        return self.yield_strength * self.area

    def get_compressive_failure_force(self) -> float:
        return self.compressive_strength * self.area

    def get_critical_buckling_load(self) -> float:
        return self.elasticity * self.moment * math.pi ** 2 / self.length ** 2

    def get_ultimate_moment(self) -> float:
        return self.yield_strength * self.area * self.radius / 2

    def get_material_properties(self):
        return {
            "elasticity": self.elasticity,
            "area": self.area,
            "moment": self.moment,
            "radius": self.radius,
            "density": self.density,
            "yield_strength": self.yield_strength,
            "compressive_strength": self.compressive_strength
        }

    def __str__(self):
        return f"[{self.start}--{self.end}]"

    def __repr__(self):
        return self.__str__()


class ElementFactory:
    elasticity: float  # Young's modulus (E)
    area: float  # Cross-sectional area (A)
    moment: float  # Second moment of area / area moment of inertia (I)
    radius: float  # Centroidal distance of cross-section (might not be half-width if asymmetric)
    density: float
    yield_strength: float
    compressive_strength: float
    flexural_strength: float

    def __init__(self, elasticity: float = None, area: float = None, density: float = None, moment: float = None,
                 radius: float = None, yield_strength: float = None, compressive_strength: float = None, flexural_strength: float = None):
        self.elasticity = elasticity if elasticity is not None else Element.default_elasticity
        self.area = area if area is not None else Element.default_area
        self.moment = moment if moment is not None else Element.default_moment
        self.radius = radius if radius is not None else Element.default_radius
        self.density = density if density is not None else Element.default_density
        self.yield_strength = yield_strength if yield_strength is not None else Element.default_yield_strength
        self.compressive_strength = compressive_strength if compressive_strength is not None else Element.default_compressive_strength
        self.flexural_strength = flexural_strength if flexural_strength is not None else Element.default_flexural_strength

    def create_element(self, start: Node, end: Node) -> Element:
        return Element(start, end, self.elasticity, self.area, self.moment, self.radius,
                       self.density, self.yield_strength, self.compressive_strength, self.flexural_strength)

    def set_cross_section_solid_square(self, width: float):
        self.area = width ** 2
        self.moment = width ** 4 / 12
        self.radius = width / 2

    def set_cross_section_hollow_square(self, width: float, thickness: float):
        inner_width = width - thickness * 2
        self.area = width ** 2 - inner_width ** 2
        self.moment = (width ** 4 - inner_width ** 4) / 12
        self.radius = width / 2

    def set_material_steel(self):
        # S355 Steel
        self.elasticity = 210 * 1000 ** 3  # 210 GPa
        self.density = 7850  # kg/m^3
        self.yield_strength = 355 * 1000 ** 2  # 355 MPa
        self.compressive_strength = 355 * 1000 ** 2  # 355 MPa
        self.flexural_strength = 355 * 1000 ** 2  # 355 MPa

    def set_material_wood(self):
        # White Oak
        # https://www.conradfp.com/pdf/ch4-Mechanical-Properties-of-Wood.pdf
        self.elasticity = 12.27 * 1000 ** 3  # 12.27 GPa
        self.density = 678  # kg/m^3
        self.yield_strength = 104.8 * 1000 ** 2  # 104.8 MPa
        self.compressive_strength = 51.3 * 1000 ** 2  # 51.3 MPa
        self.flexural_strength = 104.8 * 1000 ** 2  # 104.8 MPa

    def set_material_carbon_fiber(self):
        # http://www.performance-composites.com/carbonfibre/mechanicalproperties_2.asp
        self.elasticity = 135 * 1000 ** 3  # 135 GPa
        self.density = 1600  # kg/m^3
        self.yield_strength = 1500 * 1000 ** 2  # 1500 MPa
        self.compressive_strength = 1200 * 1000 ** 2  # 1200 MPa
        self.flexural_strength = 305 * 1000 ** 2  # 305 MPa


class Frame:
    nodes: List[Node]
    elements: List[Element]

    node_elements = Dict[Node, List[Element]]
    node_neighbors = Dict[Node, List[Element]]

    node_indices: Dict[Node, int]
    stiffness_matrix: numpy.ndarray
    _external_force_vector: Optional[numpy.ndarray]
    _displacement_vector: Optional[numpy.ndarray]
    _force_vector: Optional[numpy.ndarray]
    applied_force_vector: Optional[numpy.ndarray]

    def __init__(self, nodes: List[Node], elements: List[Element]):
        self.nodes = nodes
        self.elements = elements

        self._external_force_vector = None
        self._displacement_vector = None
        self._force_vector = None

        self.__create_graph()
        self.calculate_derived_values()

    @property
    def force_vector(self) -> numpy.ndarray:
        if self._force_vector is None:
            self.calculate_forces()
        return self._force_vector

    @force_vector.setter
    def force_vector(self, value: numpy.ndarray):
        self._force_vector = value

    @property
    def displacement_vector(self) -> numpy.ndarray:
        if self._displacement_vector is None:
            self.calculate_forces()
        return self._displacement_vector

    @displacement_vector.setter
    def displacement_vector(self, value: numpy.ndarray):
        self._displacement_vector = value

    @property
    def external_force_vector(self) -> numpy.ndarray:
        if self._external_force_vector is None:
            self.set_applied_forces()
        return self._external_force_vector

    @external_force_vector.setter
    def external_force_vector(self, value: numpy.ndarray):
        self._external_force_vector = value

    def clone(self) -> "Frame":
        return Frame(list(self.nodes), list(self.elements))

    def calculate_forces(self) -> (numpy.ndarray, numpy.ndarray):
        free_subindices = list()
        for index, node in enumerate(self.nodes):
            if not node.fixed_x:
                free_subindices.append(index * 3)
            if not node.fixed_y:
                free_subindices.append(index * 3 + 1)
            if not node.fixed_angle:
                free_subindices.append(index * 3 + 2)

        reduced_stiffness_matrix: numpy.ndarray = self.stiffness_matrix[numpy.ix_(free_subindices, free_subindices)]
        reduced_force_vector = self.external_force_vector[free_subindices]
        try:
            reduced_displacement_vector = scipy.linalg.solve(reduced_stiffness_matrix, reduced_force_vector,
                                                             assume_a="sym")
        except Exception as exception:
            eigenvalues = numpy.linalg.eigvalsh(reduced_stiffness_matrix)
            zero_eigenvalues = len([value for value in eigenvalues if abs(value) < 10 ** -5])
            if zero_eigenvalues > 0:
                # print(f"Invalid frame. Stiffness matrix condition: {log_condition}. {zero_eigenvalues} zero eigenvalues.")
                raise UnstableFrameException(zero_eigenvalues)
            else:
                raise exception

        self.displacement_vector = numpy.zeros(len(self.nodes) * 3)
        for reduced_index, index in enumerate(free_subindices):
            self.displacement_vector[index] = reduced_displacement_vector[reduced_index]

        self.force_vector = self.stiffness_matrix @ self.displacement_vector
        return self.force_vector, self.displacement_vector

    def calculate_derived_values(self):
        self.node_indices = {node: index for index, node in enumerate(self.nodes)}
        self.__build_stiffness_matrix()

    def __create_graph(self):
        self.node_elements = dict()
        self.node_neighbors = dict()
        for node in self.nodes:
            self.node_elements[node] = list()
            self.node_neighbors[node] = list()

        for element in self.elements:
            self.node_elements[element.start].append(element)
            self.node_elements[element.end].append(element)
            self.node_neighbors[element.start].append(element.end)
            self.node_neighbors[element.end].append(element.start)

    def __add_node(self, node: Node):
        self.nodes.append(node)
        self.node_elements[node] = list()
        self.node_neighbors[node] = list()

    def __add_element(self, element: Element):
        if element.end in self.node_neighbors[element.start]:
            raise ValueError("This element already exists in the frame.")
        self.elements.append(element)
        self.node_elements[element.start].append(element)
        self.node_elements[element.end].append(element)
        self.node_neighbors[element.start].append(element.end)
        self.node_neighbors[element.end].append(element.end)

    def __remove_node(self, node: Node):
        self.nodes.remove(node)
        elements_to_remove = list(self.node_elements[node])
        for element in elements_to_remove:
            self.__remove_element(element)

    def __remove_element(self, element: Element):
        self.elements.remove(element)
        self.node_elements[element.start].remove(element)
        self.node_elements[element.end].remove(element)
        self.node_neighbors[element.start].remove(element.end)
        self.node_neighbors[element.end].remove(element.start)

    def __build_stiffness_matrix(self):
        zero_submatrix = numpy.zeros((3, 3))
        submatrices = [[zero_submatrix] * len(self.nodes) for _ in range(len(self.nodes))]

        for element in self.elements:
            start_index = self.node_indices[element.start]
            end_index = self.node_indices[element.end]

            location_indices = ((start_index, start_index), (start_index, end_index),
                                (end_index, start_index), (end_index, end_index))
            sub_arrays = (element.stiffness_matrix[0:3, 0:3], element.stiffness_matrix[0:3, 3:6],
                          element.stiffness_matrix[3:6, 0:3], element.stiffness_matrix[3:6, 3:6])
            for location, array in zip(location_indices, sub_arrays):
                if submatrices[location[0]][location[1]] is zero_submatrix:
                    submatrices[location[0]][location[1]] = array
                else:
                    submatrices[location[0]][location[1]] = submatrices[location[0]][location[1]] + array
        submatrix_rows = [numpy.concatenate(row, axis=1) for row in submatrices]
        self.stiffness_matrix = numpy.concatenate(submatrix_rows, axis=0)

    def set_applied_forces(self, applied_force_vector: Optional[numpy.ndarray] = None, gravity=GRAVITATIONAL_ACCELERATION):
        self.applied_force_vector = applied_force_vector

        gravity_forces = list()
        node_weights = self.__calculate_node_weights()
        for node in self.nodes:
            gravity_forces.append(0)  # x-axis
            gravity_forces.append(node_weights[node] * -gravity)  # y-axis
            gravity_forces.append(0)  # angle-axis
        gravity_force_vector = numpy.array(gravity_forces)
        if self.applied_force_vector is not None:
            self.external_force_vector = gravity_force_vector + self.applied_force_vector
        else:
            self.external_force_vector = gravity_force_vector

    # Simplification: all weight is stored in the nodes, which get half the weight of each element.
    def __calculate_node_weights(self):
        node_weights = {node: 0 for node in self.nodes}
        for element in self.elements:
            node_weights[element.start] += element.weight / 2
            node_weights[element.end] += element.weight / 2
        return node_weights

    def build_applied_force_vector(self, node_forces: Dict[Node, Tuple[float, float, float]]) -> numpy.ndarray:
        applied_forces = list()
        for node in self.nodes:
            if node in node_forces:
                applied_forces.extend(node_forces[node])
            else:
                applied_forces.extend((0, 0, 0))
        return numpy.array(applied_forces)

    def get_element_failure_ratios(self) -> Dict[Element, float]:
        failure_ratios = dict()
        for element in self.elements:
            start_index = self.node_indices[element.start]
            end_index = self.node_indices[element.end]
            vector_indices = [start_index * 3, start_index * 3 + 1, start_index * 3 + 2, end_index * 3,
                              end_index * 3 + 1, end_index * 3 + 2]
            displacements = self.displacement_vector[vector_indices]
            failure_ratios[element] = element.get_failure_ratio(displacements)
        return failure_ratios

    def get_failed_elements(self) -> List[Element]:
        element_failure_ratios = self.get_element_failure_ratios()
        return [element for element in self.elements if element_failure_ratios[element] > 1
                or element_failure_ratios[element] < -1]

    def get_displaced_frame(self, factor: float):
        displaced_nodes = list()
        for index, node in enumerate(self.nodes):
            displaced_nodes.append(Node(node.x + factor * self.displacement_vector[index * 3],
                                        node.y + factor * self.displacement_vector[index * 3 + 1], node.fixed_x,
                                        node.fixed_y, node.fixed_angle))

        displaced_elements = list()
        for element in self.elements:
            displaced_elements.append(Element(displaced_nodes[self.node_indices[element.start]],
                                              displaced_nodes[self.node_indices[element.end]],
                                              **element.get_material_properties()))

        displaced_frame = Frame(displaced_nodes, displaced_elements)
        displaced_frame.set_applied_forces(self.applied_force_vector)
        return displaced_frame

    def prune_disconnected_components(self):
        visited_nodes = set()
        frontier = [self.nodes[0]]
        while len(frontier) > 0:
            current = frontier.pop()
            for neighbor in self.node_neighbors[current]:
                if neighbor not in visited_nodes and neighbor not in frontier:
                    frontier.append(neighbor)
            visited_nodes.add(current)
        disconnected = [node for node in self.nodes if node not in visited_nodes]
        for node in disconnected:
            self.__remove_node(node)
        self.calculate_derived_values()
        return len(disconnected)

    def prune_degree_one_components(self):
        removed_nodes = list()
        continue_search = True
        while continue_search:
            continue_search = False
            for node in self.nodes:
                if len(self.node_elements[node]) == 1:
                    if node.fixed_x or node.fixed_y:
                        continue
                    self.__remove_node(node)
                    removed_nodes.append(node)
                    continue_search = True
        self.calculate_derived_values()
        return len(removed_nodes)

    def get_node_at(self, x, y):
        for node in self.nodes:
            if node.x == x and node.y == y:
                return node
        return None

    @staticmethod
    def from_points(fixed_points: PointList, added_points: PointList, connection_distance: float,
                    element_factory: ElementFactory = None) -> "Frame":
        if element_factory is None:
            element_factory = ElementFactory()

        nodes = list()
        for x, y in fixed_points:
            nodes.append(Node(x, y, True))
        for x, y in added_points:
            nodes.append(Node(x, y))

        elements = list()
        for start_node, end_node in itertools.combinations(nodes, 2):
            if math.dist((start_node.x, start_node.y), (end_node.x, end_node.y)) <= connection_distance:
                elements.append(element_factory.create_element(start_node, end_node))

        frame = Frame(nodes, elements)
        return frame


class UnstableFrameException(Exception):
    degrees_of_freedom: int

    def __init__(self, degrees_of_freedom: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.degrees_of_freedom = degrees_of_freedom

    def __str__(self):
        return f"This frame is mathematically unstable and could not be simulated." \
               f"The frame had {self.degrees_of_freedom} degrees of freedom, and should have 0." \
               f"This should only occur if the frame is disconnected, or if it has no fixed points."


def plot_frame(frame: Frame, title: str = None, show_forces: bool = True, substitute_forces: Dict[Element, float] = None,
               plot_displacement: bool = False, displacement_factor: float = 10, filename: str = None):
    x_positions = [node.x for node in frame.nodes]
    y_positions = [node.y for node in frame.nodes]
    lines = [((element.start.x, element.start.y), (element.end.x, element.end.y)) for element in frame.elements]
    norm = pyplot.Normalize(-1, 1)
    line_collection = matplotlib.collections.LineCollection(lines, cmap="viridis", norm=norm, linewidths=5)
    if show_forces:
        if substitute_forces is not None:
            failure_ratios = substitute_forces
        else:
            failure_ratios = frame.get_element_failure_ratios()
        stress_fraction = [failure_ratios[element] for element in frame.elements]
        line_collection.set_array(stress_fraction)
    else:
        colors = [random.random() * 2 - 1 for element in frame.elements]
        line_collection.set_array(colors)
        failure_ratios = None
    figure: pyplot.Figure
    axes: pyplot.Axes
    figure, axes = pyplot.subplots()
    if title is not None:
        axes.set_title(title)
    margin = min(max(x_positions) - min(x_positions), max(y_positions) - min(y_positions)) * 0.1
    axes.set_xbound(min(x_positions) - margin, max(x_positions) + margin)
    axes.set_ybound(min(y_positions) - margin, max(y_positions) + margin)
    axes.set_aspect("equal")
    # axes.margins(0.5)
    # workaround for matplotlib bug
    line_collection.__dict__['_A'] = numpy.array(line_collection.__dict__['_A'])
    plot_lines = axes.add_collection(line_collection)
    figure.colorbar(plot_lines, ax=axes)
    if filename is not None:
        figure.savefig(f"Plots/{filename}")
    else:
        figure.show()
    if plot_displacement:
        displaced_frame = frame.get_displaced_frame(displacement_factor)
        displaced_forces = None
        if show_forces:
            displaced_forces = {displaced_frame.elements[i]: failure_ratios[element] for i, element in enumerate(frame.elements)}
        displaced_title = None
        if title is not None:
            displaced_title = f"{title}\n(Displacement × {displacement_factor})"
        displaced_filename = None
        if filename is not None:
            filename_parts = filename.split(".")
            filename_start = ".".join(filename_parts[:-1])
            displaced_filename = f"{filename_start}_displaced.{filename_parts[-1]}"
        plot_frame(displaced_frame, title=displaced_title, show_forces=show_forces, substitute_forces=displaced_forces,
                   filename=displaced_filename)


class EvaluateFrameResult(TypedDict):
    frame: Frame
    """The frame evaluated, with forces shown for its maximum load."""
    weight: float
    """The maximum load weight of the frame, in Newtons."""
    repaired_disconnected: int
    """The number of repair operations that deleted disconnected elements of the frame."""
    gravity_reduction: float
    """The percentage of gravity reduction needed for the frame to support its own weight."""
    invalid: bool
    """If this is true, the frame had a fault that made it impossible to calculate fitness."""


def evaluate_frame(fixed_points: PointList, load_points: PointList, input_points: PointList,
                   connection_distance: float, weight_resolution: int = 100, relax_gravity: bool = False,
                   gravity_resolution: float = 0.01, element_factory: ElementFactory = None, **kwargs) -> EvaluateFrameResult:
    """
    Build a frame from the given parameters, and evaluate the maximum load that can be placed on the load points.
    Values are given in SI units (meters, Newtons, etc.).

    Args:
        fixed_points: A list of (x, y) coordinates for immovable anchor points.
        load_points: A list of (x, y) coordinates where load will be distributed (e.g. a roadbed).
        input_points: A list of (x, y) coordinates to build the rest of the bridge out of.
        connection_distance: The maximum distance at which to connect two points.
        weight_resolution: The resolution at which to slowly increase weight until the frame breaks.
            Affects speed substantially, as this defines the depth of the binary search!
        relax_gravity: Whether to try reducing the weight of the frame if it can't hold itself.
        gravity_resolution: The resolution at which to slowly decrease gravity until the frame doesn't break.
            Affects speed substantially, as this defines the depth of the binary search!
        element_factory: An ElementFactory configured with the material properties that elements of the frame should
            be given. I think this is how factories work?

    Returns: A dictionary of properties of the frame matching the EvaluateFrameResult type.
        These should be used to calculate fitness as desired.

    Todo: Consider other metrics, like the average stress on frame elements.
    """
    added_points = set([tuple(element) for element in load_points])
    added_points.update([tuple(element) for element in input_points])
    added_points = list(added_points)
    if element_factory is None:
        element_factory = ElementFactory()
    frame = Frame.from_points(fixed_points, added_points, connection_distance, element_factory)
    pruned_disconnected = frame.prune_disconnected_components()

    load_nodes = list()
    for point in load_points:
        node = frame.get_node_at(*point)
        load_nodes.append(node)

    results: EvaluateFrameResult = {
        "frame": frame,
        "weight": 0,
        "repaired_disconnected": pruned_disconnected,
        "gravity_reduction": 0,
        "invalid": True
    }

    try:
        weight_generator = weight_search_generator(weight_resolution)
        weight = next(weight_generator)
        while True:
            failures = __evaluate_frame_weight(frame, load_nodes, weight)
            if len(failures) > 0:
                weight = weight_generator.send(True)
            else:
                weight = weight_generator.send(False)
    except StopIteration as stop_iteration:
        failure_weight = stop_iteration.value
    except numpy.LinAlgError:
        results["invalid"] = True
        return results
    except:
        results["invalid"] = True
        return results

    if failure_weight == 0 and relax_gravity:
        results["invalid"] = False
        try:
            gravity_generator = gravity_search_generator(gravity_resolution)
            gravity_factor = next(gravity_generator)
            while True:
                failures = __evaluate_frame_weight(frame, load_nodes, 0, gravity_factor)
                if len(failures) > 0:
                    gravity_factor = gravity_generator.send(True)
                else:
                    gravity_factor = gravity_generator.send(False)
        except StopIteration as stop_iteration:
            results["gravity_reduction"] = 1 - stop_iteration.value
        except numpy.LinAlgError:
            results["invalid"] = True
            return results
        except:
            results["invalid"] = True
            return results
    elif failure_weight == 0:
        results["invalid"] = True
    else:
        results["weight"] = failure_weight - weight_resolution
        results["invalid"] = False
    results["frame"] = frame
    return results


def __evaluate_frame_weight(frame: Frame, active_load_nodes: List[Node], weight: float, gravity_factor: float = 1) -> List[Element]:
    weight_per_point = weight / len(active_load_nodes)
    load = {node: (0, -weight_per_point, 0) for node in active_load_nodes}
    applied_force_vector = frame.build_applied_force_vector(load)
    frame.set_applied_forces(applied_force_vector, gravity=GRAVITATIONAL_ACCELERATION * gravity_factor)
    frame.calculate_forces()
    failures = frame.get_failed_elements()
    return failures


def weight_search_generator(resolution: float):
    """Finds the lowest weight at which the bridge fails."""
    failed = yield 0
    if failed:
        return 0
    search_value = 1
    while not failed:
        failed = yield search_value * resolution
        if not failed:
            search_value *= 2
    upper_bound = search_value
    lower_bound = search_value // 2 + 1
    while upper_bound > lower_bound:
        # Use floor so as not to search the upper bound (which is known to fail)
        search_value = math.floor((upper_bound + lower_bound) / 2)
        failed = yield search_value * resolution
        if failed:
            # Failed, so include search value
            upper_bound = search_value
        else:
            # Did not fail, so exclude search value
            lower_bound = search_value + 1
    return upper_bound * resolution


def gravity_search_generator(resolution: float):
    """Finds the lowest gravity reduction at which the bridge does not fail."""
    if resolution >= 1 or resolution <= 0:
        raise ValueError("Resolution must be between 0 and 1.")
    upper_bound = int(1 / resolution)
    lower_bound = 0
    while upper_bound > lower_bound:
        # Use ceiling so as not to search 0 (which obviously succeeds)
        search_value = math.ceil((upper_bound + lower_bound) / 2)
        failed = yield search_value * resolution
        if failed:
            # Failed, so exclude search value
            upper_bound = search_value - 1
        else:
            # Did not fail, so include search value
            lower_bound = search_value
    return upper_bound * resolution


if __name__ == "__main__":
    # """
    span = 300
    fixed_points = [(0, 0), (span, 0)]
    # added_points = [(x, 10 * math.sin(x * math.pi / 100)) for x in range(10, 100, 10)]
    load_points = [(x, 0) for x in range(20, span, 20)]
    added_points = [(x, 15) for x in range(10, span, 20)]
    element_factory = ElementFactory()
    #element_factory.set_cross_section_hollow_square(0.4, 0.04)
    element_factory.set_material_wood()
    element_factory.set_cross_section_solid_square(0.125)
    frame = Frame.from_points(fixed_points, added_points, 20, element_factory)
    result = evaluate_frame(fixed_points, load_points, added_points, 20, element_factory=element_factory)
    frame = result["frame"]
    # frame.prune_disconnected_components()
    # frame.prune_degree_one_components()
    # frame.set_applied_forces()
    # frame.calculate_forces()
    plot_frame(frame, title=f"Failure load: {result['weight']} N", plot_displacement=True, filename="../ea/Plots/test.png")
    #displaced_frame = frame.get_displaced_frame(5)
    #displaced_frame.set_applied_forces()
    #plot_frame(displaced_frame)
    failed_elements = frame.get_failed_elements()
    print(failed_elements)
    pass

    """
    # https://elib.unikom.ac.id/files/disk1/379/jbptunikompp-gdl-ydjokoseti-18942-14-14_matri-e.pdf example 1
    A = Node(0, 6, False, True, False)
    B = Node(6, 6, False)
    C = Node(6, 0, True)
    element_factory = ElementFactory(elasticity=200*10**6, moment=60*10**-6, area=600*10**-6)
    AB = element_factory.create_element(A, B)
    BC = element_factory.create_element(B, C)
    frame = Frame([A, B, C], [AB, BC])
    applied_forces = frame.build_applied_force_vector({B: (5, 0, 0)})
    frame.set_applied_forces(applied_forces, gravity=0)
    frame.calculate_forces()
    element_stresses = frame.get_element_axial_stresses()
    pass
    """