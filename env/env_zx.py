import os
import math
import gymnasium as gym
import networkx as nx
import pyzx as zx
import random
from typing import List, Tuple, Union, Optional, Callable
import re
import torch
from gymnasium import spaces
import numpy as np
from env.circuit_utils import load_gate_sequence_from_txt, load_gate_sequence_from_qcis, \
    gate_sequence_to_tensor
from pyzx.graph import VertexType, EdgeType
from pyzx.circuit import Circuit
from fractions import Fraction
from sympy import sympify
from torch_geometric.data import Data
import qiskit.qasm2
from qiskit import QuantumCircuit, qasm2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

GateSeq = List[Tuple[str, Tuple[str, ...], Union[float, None]]]


def qcis_to_qasm(gate_sequence: List[Tuple[str, List[str], Union[float, int, None]]]) -> str:
    """
    Convert a QCIS instruction set to QASM format.

    Parameters:
    - gate_sequence: A list of tuples containing (gate, quantum bits, parameter)

    Returns:
    - qasm_str: A string in QASM format
    """
    qasm_str = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n"

    # Find the highest quantum bit index to determine the size of the quantum register
    max_qubit = 0

    for _, qubits, _ in gate_sequence:
        max_qubit = max(max_qubit, *[int(q[1:]) for q in qubits])

    # Define the quantum register
    qasm_str += f"qreg q[{max_qubit + 1}];\n"
    
    # Convert each QCIS gate to QASM
    for gate, qubits, param in gate_sequence:
        qubits_str = ", ".join([f"q[{q[1:]}]" for q in qubits])

        if gate in {"X", "Y", "Z", "H"}:
            qasm_str += f"{gate.lower()} {qubits_str};\n"
        elif gate in {"RX", "RY", "RZ"} and param is not None:
            qasm_str += f"{gate.lower()}({param}) {qubits_str};\n"
        elif gate in {"X2P", "X2M", "Y2P", "Y2M"}:
            # Approximate these gates as specific rotations of RX or RY
            if gate == "X2P":
                qasm_str += f"rx(pi/4) {qubits_str};\n"
            elif gate == "X2M":
                qasm_str += f"rx(-pi/4) {qubits_str};\n"
            elif gate == "Y2P":
                qasm_str += f"ry(pi/4) {qubits_str};\n"
            elif gate == "Y2M":
                qasm_str += f"ry(-pi/4) {qubits_str};\n"
        elif gate in {"S", "Sdg", "T", "Tdg"}:
            if gate == "S":
                qasm_str += f"s {qubits_str};\n"
            elif gate == "Sdg":
                qasm_str += f"sdg {qubits_str};\n"
            elif gate == "T":
                qasm_str += f"t {qubits_str};\n"
            elif gate == "Tdg":
                qasm_str += f"tdg {qubits_str};\n"
        elif gate == "CZ":
            qasm_str += f"cz {qubits_str};\n"
        

    original_qubits = [f"Q{i}" for i in range(max_qubit + 1)]
    return qasm_str, original_qubits


class QuantumCircuitSimplificationEnv(gym.Env):
    def __init__(self, chip_graph: nx.Graph):
        super(QuantumCircuitSimplificationEnv, self).__init__()

        self.chip = chip_graph
        self.special_vertices = []
        self.input_tensor = None  # Initialize a tensor to store generated data
        self.priority_gates = {"T", "S", "SD", "TD", "CZ", "H", "RX", "RY", "RZ", "X2P", "X2M", "Y2P", "Y2M"}

        # Initialize file paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.input_folder = os.path.abspath(os.path.join(self.base_dir, '..', 'data', 'Dataset'))
        self.output_folder = os.path.abspath(os.path.join(self.base_dir, '..', 'data', 'output'))
        os.makedirs(self.output_folder, exist_ok=True)  # Ensure the output folder exists

        # Read the list of files
        self.all_files = sorted(os.listdir(self.input_folder))

        self.graph = None  # Initialize the circuit graph
        self.physical_gate_sequence = []
        self.original_sequence = []
        self.current_gate_index = 0

        # Initialize the feature dimensions of vertices and edges, adjusted based on node and edge types in the gate sequence
        self.node_feature_dim = 10  # Node feature dimension, e.g., type, phase, etc.
        self.edge_feature_dim = 3  # Edge feature dimension, e.g., type, weight, etc.

        # Define action space and observation space
        self.action_space = spaces.Discrete(len(chip_graph.nodes))  # Node selection
        self.observation_space = spaces.Dict({
            "x": spaces.Box(low=0, high=1, shape=(len(chip_graph.nodes), self.node_feature_dim), dtype=np.float32),
            "edge_index": spaces.Box(low=0, high=len(chip_graph.nodes), shape=(2, len(chip_graph.edges)),
                                     dtype=np.int64),
            "edge_attr": spaces.Box(low=0, high=1, shape=(len(chip_graph.edges), self.edge_feature_dim),
                                    dtype=np.float32),
            "y": spaces.Discrete(len(chip_graph.nodes))  # Initialize action encoding for y
        })

    def _evolution(self):
        print("[DEBUG] Simplifying the circuit before evolution starts.")
        self.simplify_circuit()
        print("[DEBUG] Start evolution process.")

        vertices_list = list(self.zx_graph.vertices())

        for vertex in vertices_list:
            try:
                # Check if the vertex still exists in the circuit
                if vertex not in self.zx_graph.vertices():
                    print(f"[ERROR] Vertex {vertex} no longer exists in the graph.")
                    continue

                qubits = self.get_qubits(vertex)

                if not qubits:
                    qubits = [0]  # Assign a default quantum bit for isolated vertices

                # Handle the logic for exploring isolated vertices
                if self.is_isolated_vertex(vertex):
                    self.explore_isolated_vertex(vertex, qubits)
                else:
                    gate = self.map_gate(vertex)
                    if gate:
                        self.physical_gate_sequence.append((gate, qubits, None))
                    else:
                        print(f"[WARNING] Failed to map a gate for vertex {vertex}, gate mapping returned None.")

            except KeyError as e:
                print(f"[ERROR] KeyError encountered for vertex {vertex}: {e}. Skipping this vertex.")
                continue

            self.current_gate_index += 1
            yield

    def reset(self, file_path=None):
        """
        Reset the environment state to ensure consistency.
        """
        # Initialize variables related to stagnation
        self.stagnant_epochs = 0
        self.history = []  # Initialize reward history

        if file_path is None:
            # Randomly select a file from the file list
            input_file = random.choice(self.all_files)
            file_path = os.path.join(self.input_folder, input_file)
        else:
            # If a file path is provided, extract the file name
            input_file = os.path.basename(file_path)

        # Load the QASM file and convert it directly to a zx.Graph object
        if input_file.endswith('.qasm'):
            self.circuit = zx.Circuit.load(file_path)  # Load the QASM circuit
            self.zx_graph = self.circuit.to_graph()
            self.original_circuit = zx.Circuit.load(file_path)
            self.original_zx_graph = self.original_circuit.to_graph()
        
        # Load the quantum gate sequence based on the file type
        if input_file.endswith('.txt'):
            gate_sequence = load_gate_sequence_from_txt(file_path)
            # Convert the gate sequence to a zx.Graph object
            self.qasm_str, self.original_qubits = qcis_to_qasm(gate_sequence)
            self.circuit = Circuit.from_qasm(self.qasm_str)
            self.qasm_str1 = self.circuit.to_qasm()
            self.qc = QuantumCircuit.from_qasm_str(self.qasm_str1)
            self.pm = generate_preset_pass_manager(optimization_level=3, basis_gates=["x", "y", "z", "h", "rx", "ry", "rz", "cz", "sx", "sxdg", "s", "sdg", "t", "tdg"])
            self.qc_trans = self.pm.run(self.qc)
            self.qasm_str2 = qasm2.dumps(self.qc_trans)
            self.circuit2 = Circuit.from_qasm(self.qasm_str2)
            self.zx_graph = self.circuit2.to_graph()
            self.original_gate_sequence = gate_sequence
            self.original_qasm_str, self.strat__qubits = qcis_to_qasm(self.original_gate_sequence)
            self.original_circuit = Circuit.from_qasm(self.original_qasm_str)
            self.qasm_str3 = self.original_circuit.to_qasm()
            self.qc2 = QuantumCircuit.from_qasm_str(self.qasm_str3)
            self.pm2 = generate_preset_pass_manager(optimization_level=3, basis_gates=["x", "y", "z", "h", "rx", "ry", "rz", "cz", "sx", "sxdg", "s", "sdg", "t", "tdg"])
            self.qc_trans2 = self.pm2.run(self.qc2)
            self.qasm_str4 = qasm2.dumps(self.qc_trans2)
            self.original_circuit2 = Circuit.from_qasm(self.qasm_str4)
            self.original_zx_graph = self.original_circuit2.to_graph()
        elif input_file.endswith('.qcis'):
            gate_sequence = load_gate_sequence_from_qcis(file_path)
            self.qasm_str, self.original_qubits = qcis_to_qasm(gate_sequence)
            self.circuit = Circuit.from_qasm(self.qasm_str)
            self.qasm_str1 = self.circuit.to_qasm()
            self.qc = QuantumCircuit.from_qasm_str(self.qasm_str1)
            self.pm = generate_preset_pass_manager(optimization_level=3, basis_gates=["x", "y", "z", "h", "rx", "ry", "rz", "cz", "sx", "sxdg", "s", "sdg", "t", "tdg"])
            self.qc_trans = self.pm.run(self.qc)
            self.qasm_str2 = qasm2.dumps(self.qc_trans)
            self.circuit2 = Circuit.from_qasm(self.qasm_str2)
            self.zx_graph = self.circuit2.to_graph()
            self.original_gate_sequence = gate_sequence
            self.original_qasm_str, self.strat__qubits = qcis_to_qasm(self.original_gate_sequence)
            self.original_circuit = Circuit.from_qasm(self.original_qasm_str)
            self.qasm_str3 = self.original_circuit.to_qasm()
            self.qc2 = QuantumCircuit.from_qasm_str(self.qasm_str3)
            self.pm2 = generate_preset_pass_manager(optimization_level=3, basis_gates=["x", "y", "z", "h", "rx", "ry", "rz", "cz", "sx", "sxdg", "s", "sdg", "t", "tdg"])
            self.qc_trans2 = self.pm2.run(self.qc2)
            self.qasm_str4 = qasm2.dumps(self.qc_trans2)
            self.original_circuit2 = Circuit.from_qasm(self.qasm_str4)
            self.original_zx_graph = self.original_circuit2.to_graph()

        # Get the number of nodes and edges
        num_real_nodes = self.zx_graph.num_vertices()
        num_action_nodes = 14
        total_nodes = num_real_nodes + num_action_nodes
        num_edges = self.zx_graph.num_edges()

        # Define the expected dimension of node features, ensuring it is 14 dimensions
        target_feature_dim = 14

        # 1. Initialize the node feature tensor, padding the insufficient features with 0
        node_features = np.zeros((total_nodes, target_feature_dim))  # (a, target_feature_dim)
        for i, node in enumerate(self.zx_graph.vertices()):
            node_type = self.get_node_type(node)
            phase = self.get_node_phase(node)
            gate_type = self.get_gate_type(node)
            # Fill in the existing features of the nodes
            node_features[i][:3] = self.one_hot_encode(node_type, 3)  # Node type
            node_features[i][3] = phase  # Phase
            node_features[i][4:10] = self.one_hot_encode(gate_type, 6)  # Quantum gate type
            # Remaining features are already padded with 0

        # Generate features for 14 action nodes, with the remaining features default to 0
        for i in range(num_real_nodes, total_nodes):
            node_features[i][:3] = self.one_hot_encode(3, 3)  # Action node type
            node_features[i][3] = 0  # Phase for action nodes is 0
            # Remaining features are already padded with 0

        # Process the generation of edge features and edge indices
        edge_indices = np.array([list(edge) for edge in self.zx_graph.edges()]).T  # (2, c)

        edge_attributes = np.zeros((num_edges, self.edge_feature_dim))  # (c, d)
        for i, edge in enumerate(self.zx_graph.edges()):
            edge_type = self.get_edge_type(edge)
            edge_attributes[i][:2] = self.one_hot_encode(edge_type, 2)  # Edge type
            edge_attributes[i][2] = self.get_edge_weight(edge)  # Edge weight

        # Initialize identifier and modify its structure to a 1D tensor
        y_values = np.full(num_real_nodes, -1, dtype=int)
        action_identifiers = np.arange(1, num_action_nodes + 1)
        y_values = np.concatenate([y_values, action_identifiers])  # Generate a 1D array

        # Convert to PyTorch tensors
        x_tensor = torch.tensor(node_features, dtype=torch.float32)  # (a, target_feature_dim)
        edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long)  # (2, c)
        edge_attr_tensor = torch.tensor(edge_attributes, dtype=torch.float32)  # (c, d)
        identifier_tensor = torch.tensor(y_values, dtype=torch.long)  # (a,)

        # Return a Data object containing these tensors
        data = Data(
            x=x_tensor,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor,
            identifiers=identifier_tensor
        )

        self.physical_gate_sequence = []
        self.original_sequence = []
        self.current_gate_index = 0
        return data, {}


    def step(self, action: int):
        try:
            # Decode the action to determine the operation type and the target nodes
            operation, node_list = self.decode_action(action)

            # Execute the corresponding simplification rule based on the decoded operation
            valid_reduction = False
            while not valid_reduction:
                if operation == 1:  # Clifford simplification
                    zx.simplify.clifford_simp(self.zx_graph)
                elif operation == 2:  # Pivot simplification
                    zx.simplify.pivot_simp(self.zx_graph)
                elif operation == 3:  # Spider simplification
                    zx.simplify.spider_simp(self.zx_graph)
                elif operation == 4:  # Phase gadget simplification
                    zx.simplify.gadget_simp(self.zx_graph)
                elif operation == 5:  # Full reduction
                    zx.simplify.full_reduce(self.zx_graph)
                elif operation == 6:  # Bialgebra simplification
                    zx.simplify.bialg_simp(self.zx_graph)
                elif operation == 7:  # Remove identity nodes
                    zx.simplify.id_simp(self.zx_graph)
                elif operation == 8:  # Supplementarity simplification
                    zx.simplify.supplementarity_simp(self.zx_graph)
                elif operation == 9:  # Local complementation
                    zx.simplify.lcomp_simp(self.zx_graph)
                elif operation == 10:  # Phase-free simplification
                    zx.simplify.phase_free_simp(self.zx_graph)
                elif operation == 11:  # Pivot gadget simplification
                    zx.simplify.pivot_gadget_simp(self.zx_graph)
                elif operation == 12:  # Pivot boundary simplification
                    zx.simplify.pivot_boundary_simp(self.zx_graph)
                elif operation == 13:  # Reduce scalar
                    zx.simplify.reduce_scalar(self.zx_graph)
                elif operation == 14:  # Teleport reduce
                    zx.simplify.teleport_reduce(self.zx_graph)
                
                # Check if the simplified circuit structure is valid
                valid_reduction = self.is_valid_circuit_structure()

                # If the circuit structure is invalid, choose a new simplification rule
                if not valid_reduction:
                    print("[INFO] Simplified circuit structure is not valid. Retrying with a different rule.")
                    operation = random.randint(1, 14)

            # Calculate the reward
            reward = self.calculate_reward()

            # Check if the process is complete (e.g., when no further simplifications can be applied)
            done = self.is_terminal_state()

            # Generate a new observation
            observation = self._get_observation()

            return observation, reward, done, False, {}

        except StopIteration:
            # When the simplification process is complete, save the results and return the final reward
            output_path = os.path.join(self.output_folder, 'simplified_circuit.txt')
            self.save_physical_gate_sequence_to_txt(output_path)

            reward = self.calculate_reward()
            return self._get_observation(), reward, True, False, {}

    def is_valid_circuit_structure(self):
        """
        Check if the simplified circuit structure is valid.
        :return: True if the circuit structure is valid, otherwise False
        """
        # The validity can be determined based on the topology and logical relationships of the circuit.
        # For example, check if all quantum bits have corresponding operations, and all necessary edges exist.
        return self.zx_graph.num_edges() > 0 and self.zx_graph.num_vertices() > 1



    def _get_observation(self):
        # Get the number of nodes and edges in the current ZX graph
        num_real_nodes = self.zx_graph.num_vertices()
        num_action_nodes = 14
        total_nodes = num_real_nodes + num_action_nodes
        num_edges = self.zx_graph.num_edges()

        # Define the expected node feature dimension, ensuring it is 14 dimensions
        target_feature_dim = 14

        # 1. Initialize the node feature tensor, padding insufficient features with 0
        node_features = np.zeros((total_nodes, target_feature_dim))  # (a, target_feature_dim)
        for i, node in enumerate(self.zx_graph.vertices()):
            node_type = self.get_node_type(node)
            phase = self.get_node_phase(node)
            gate_type = self.get_gate_type(node)
            # Fill in the existing features of the nodes
            node_features[i][:3] = self.one_hot_encode(node_type, 3)  # Node type
            node_features[i][3] = phase  # Phase
            node_features[i][4:10] = self.one_hot_encode(gate_type, 6)  # Quantum gate type
            # Remaining features are already padded with 0

        # Generate features for 14 action nodes, with the remaining features defaulting to 0
        for i in range(num_real_nodes, total_nodes):
            node_features[i][:3] = self.one_hot_encode(3, 3)  # Action node type
            node_features[i][3] = 0  # Phase for action nodes is 0
            # Remaining features are already padded with 0

        # Process the generation of edge features and edge indices
        edge_indices = np.array([list(edge) for edge in self.zx_graph.edges()]).T  # (2, c)

        edge_attributes = np.zeros((num_edges, self.edge_feature_dim))  # (c, d)
        for i, edge in enumerate(self.zx_graph.edges()):
            edge_type = self.get_edge_type(edge)
            edge_attributes[i][:2] = self.one_hot_encode(edge_type, 2)  # Edge type
            edge_attributes[i][2] = self.get_edge_weight(edge)  # Edge weight

        # Initialize the identifier and modify its structure to a 1D tensor
        y_values = np.full(num_real_nodes, -1, dtype=int)
        action_identifiers = np.arange(1, num_action_nodes + 1)
        y_values = np.concatenate([y_values, action_identifiers])  # Generate a 1D array

        # Convert to PyTorch tensors
        x_tensor = torch.tensor(node_features, dtype=torch.float32)  # (a, target_feature_dim)
        edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long)  # (2, c)
        edge_attr_tensor = torch.tensor(edge_attributes, dtype=torch.float32)  # (c, d)
        identifier_tensor = torch.tensor(y_values, dtype=torch.long)  # (a,)

        # Return a Data object containing these tensors
        return Data(
            x=x_tensor,  # (a, target_feature_dim)
            edge_index=edge_index_tensor,  # (2, c)
            edge_attr=edge_attr_tensor,  # (c, d)
            identifiers=identifier_tensor  # (a,)
        )

    # Add a new function to generate the input tensor
    def generate_input_tensor(self, gate_sequence):
        """
        Generate an input tensor based on the quantum gate sequence.
        """
        max_float_index = max(float(qubit[1:]) for gate_tuple in gate_sequence for qubit in gate_tuple[1])
        qubit_count_int = int(math.ceil(max_float_index)) + 1  # Calculate the size of the tensor
        return gate_sequence_to_tensor(gate_sequence, qubit_count_int)

    def is_terminal_state(self):
        """
        Check if the current circuit cannot be further simplified.
        A terminal state is reached when the number of vertices and edges in the circuit no longer changes,
        or when no further simplification operations can be performed.
        """
        current_vertex_count = self.zx_graph.num_vertices()
        current_edge_count = self.zx_graph.num_edges()

        # Get the current number of vertices and edges in the circuit
        if not hasattr(self, 'previous_vertex_count'):
            self.previous_vertex_count = current_vertex_count
        if not hasattr(self, 'previous_edge_count'):
            self.previous_edge_count = current_edge_count

        # If the number of vertices and edges has not changed, the circuit cannot be further simplified
        if current_vertex_count == self.previous_vertex_count and current_edge_count == self.previous_edge_count:
            return True

        # Update the recorded number of vertices and edges
        self.previous_vertex_count = current_vertex_count
        self.previous_edge_count = current_edge_count

        # Check if there are any applicable simplification operations
        simplifiable = False

        # Try applying each simplification rule; if any rule succeeds, the circuit can still be simplified
        for simplify_func in [
            zx.simplify.clifford_simp,
            zx.simplify.pivot_simp,
            zx.simplify.spider_simp,
            zx.simplify.gadget_simp,
            zx.simplify.full_reduce,
            zx.simplify.bialg_simp,
            zx.simplify.id_simp,
            zx.simplify.supplementarity_simp,
            zx.simplify.lcomp_simp,
            zx.simplify.phase_free_simp,
            zx.simplify.pivot_gadget_simp,
            zx.simplify.pivot_boundary_simp,
            zx.simplify.reduce_scalar,
            zx.simplify.teleport_reduce,
        ]:
            temp_graph = self.zx_graph.copy()  # Use a copy for testing simplifications
            simplify_func(temp_graph, quiet=True)
            if temp_graph.num_vertices() < current_vertex_count or temp_graph.num_edges() < current_edge_count:
                simplifiable = True
                break

        if not simplifiable:
            return True

        return False

    def encode_action(self, operation: int, node_list: List[int]) -> int:
        """
        Encode a simplification operation and a list of nodes into an integer y.
        operation: Type of operation
        node_list: List of nodes
        """
        interval = 1000  # Define the range for each operation
        node_value = sum([n * (interval // (len(node_list) + 1)) for n in node_list])  # Generate a value based on the nodes
        encoded_y = operation * interval + node_value  # Combine into the final y value
        return encoded_y

    def decode_action(self, y_value: int) -> Tuple[int, List[int]]:
        """
        Decode a y value into an operation and a list of nodes.
        y_value: Encoded y value
        Returns: Type of operation, list of nodes
        """
        interval = 1000  # Define the range
        operation = y_value // interval  # Get the type of operation
        node_value = y_value % interval  # Get the value related to nodes

        node_list = []
        while node_value > 0:
            node_list.append(node_value % (interval // 2))
            node_value //= (interval // 2)

        return operation, node_list

    def one_hot_encode(self, value, num_classes):
        """
        Perform one-hot encoding for a category value, returning a vector of length num_classes.

        Parameters:
        - value: The category (integer) to be encoded.
        - num_classes: Total number of categories.

        Returns:
        - one_hot: The one-hot encoded vector.
        """
        # Initialize an array of zeros
        one_hot = [0] * num_classes

        # Ensure the input category value is within the valid range
        if 0 <= value < num_classes:
            one_hot[value] = 1  # Set the corresponding category position to 1

        return one_hot

    def get_node_type(self, node):
        """
        Get the type of the node, such as Z, X, Boundary, etc.
        """
        # Return one-hot encoding based on the vertex type in the ZX graph
        if self.zx_graph.type(node) == zx.VertexType.Z:
            return 0  # Z-type vertex
        elif self.zx_graph.type(node) == zx.VertexType.X:
            return 1  # X-type vertex
        elif self.zx_graph.type(node) == zx.VertexType.BOUNDARY:
            return 2  # Boundary vertex
        else:
            return 3  # Other types of vertices

    def get_node_phase(self, node):
        """
        Get the phase information of the node, normalized to the range [-π, π].
        """
        phase = self.zx_graph.phase(node)
        if isinstance(phase, (int, float, Fraction)):
            phase = ((float(phase) + np.pi) % (2 * np.pi)) - np.pi
        return phase

    def get_gate_type(self, node):
        """
        Get the type of quantum gate represented by the node.
        """
        if self.zx_graph.type(node) == zx.VertexType.Z:
            # Assume the quantum gate is mapped based on the phase of the vertex in the ZX graph
            phase = self.zx_graph.phase(node)
            if phase == np.pi / 2:
                return 4  # S gate
            elif phase == np.pi / 4:
                return 5  # T gate
            elif phase == -np.pi / 2:
                return 6  # SD gate
            else:
                return 7  # Other gates
        return 3  # Default to other types

    def get_edge_type(self, edge):
        """
        Get the type of the edge: regular edge or Hadamard edge.
        """
        if self.zx_graph.edge_type(edge) == zx.EdgeType.HADAMARD:
            return 1  # Hadamard edge
        else:
            return 0  # Regular edge

    def get_edge_weight(self, edge):
        """
        Design the weight of the edge based on the quantum gate instruction set 
        and the quantum bits connected by the vertices.
        """
        # Get the two vertices (quantum bits)
        node1, node2 = edge

        # Parse the type of quantum gate and set the weight based on the connection
        gate_type = self.get_gate_type_between_nodes(node1, node2)

        # Assign weights based on the type of gate
        if gate_type == "CZ":
            return 5  # Control-Z gate, assign a high weight
        elif gate_type in {"RX", "RY", "RZ"}:
            return 2  # Rotation gates, assign a lower weight
        elif gate_type == "H":
            return 3  # Hadamard gate, assign a medium weight
        else:
            return 1  # Other types of gates, default weight

    def get_gate_type_between_nodes(self, node1, node2):
        """
        Get the type of quantum gate between two quantum bits.
        """
        # Use to_qasm() to obtain a QASM format string
        qasm_str = self.original_circuit.to_qasm()

        # Iterate through each line of the QASM string
        for line in qasm_str.splitlines():
            # Skip lines that are declarations or non-gate operations
            if line.startswith("qreg") or line.startswith("creg") or line.startswith("//") or line == "":
                continue

            # Parse the quantum gate operation in each line
            parts = line.split()
            gate_type = parts[0]  # Get the type of the gate
            qubits_str = parts[1].replace(";", "")  # Remove the trailing semicolon

            # Extract the quantum bit indices
            qubits = [int(match.group(1)) for match in re.finditer(r"q\[(\d+)\]", qubits_str)]

            # If there are two quantum bits that match node1 and node2, return the corresponding gate type
            if len(qubits) == 2 and (
                    (qubits[0] == node1 and qubits[1] == node2) or (qubits[0] == node2 and qubits[1] == node1)):
                return gate_type.upper()  # Return the gate type in uppercase

        return None  # Return None if no matching gate is found
        
    
    def calculate_reward(self, discount_factor=0.99, exploration_weight=0.5):
        # Weight parameters
        weight_gate_count = 0.5  # Weight for the number of gates, reducing gate count earns higher rewards
        weight_circuit_depth = 1.0  # Weight for circuit depth, reducing depth earns higher rewards
        weight_cnot_count = 0.7  # Weight for the number of CNOT gates
        weight_t_count = 0.3  # Weight for the number of T gates

        # Physical gate sequence of the original ZX graph
        self.original_sequence = self.extract_gates_directly(self.original_zx_graph)
        # Physical gate sequence of the updated ZX graph
        self.physical_gate_sequence = self.extract_gates_directly(self.zx_graph)

        # Calculate new metrics
        gate_count = len(self.physical_gate_sequence.splitlines())  # Number of gates
        circuit_depth = self.calculate_circuit_depth(self.zx_graph)
        cnot_count, t_count, _ = self.calculate_gate_metrics_from_sequence(self.physical_gate_sequence)

        # Calculate metrics for the initial state (for comparison)
        original_gate_count = len(self.original_sequence.splitlines())
        original_circuit_depth = self.calculate_circuit_depth(self.original_zx_graph)
        original_cnot_count, original_t_count, _ = self.calculate_gate_metrics_from_sequence(self.original_sequence)

        # Reward calculation logic
        gate_count_reward = weight_gate_count * gate_count
        if gate_count > original_gate_count:  # If gate count increases after simplification, convert to a penalty
            gate_count_reward = -abs(gate_count_reward)

        circuit_depth_reward = weight_circuit_depth * circuit_depth
        if circuit_depth > original_circuit_depth:  # If circuit depth increases after simplification, convert to a penalty
            circuit_depth_reward = -abs(circuit_depth_reward)

        cnot_count_reward = weight_cnot_count * cnot_count
        if cnot_count > original_cnot_count:  # If CNOT gate count increases after simplification, convert to a penalty
            cnot_count_reward = -abs(cnot_count_reward)

        t_count_reward = weight_t_count * t_count
        if t_count > original_t_count:  # If T gate count increases after simplification, convert to a penalty
            t_count_reward = -abs(t_count_reward)

        # Exploration reward calculation
        exploration_bonus = exploration_weight * random.uniform(0, 1.0)  # Random exploration reward, range [0, 1.0]
        
        # Current reward calculation
        immediate_reward = (gate_count_reward +
                            circuit_depth_reward +
                            cnot_count_reward +
                            t_count_reward +
                            exploration_bonus)  # Includes exploration reward

        # Accumulate future rewards using a discount factor
        discounted_reward = 0
        for i, (future_reward, _) in enumerate(self.history):
            discounted_reward += (discount_factor ** i) * future_reward

        # Final reward includes both current and future discounted rewards
        total_reward = immediate_reward + discounted_reward
        return total_reward

        
    def extract_gates_directly(self, zx_graph):
        """
        Extract the physical gate sequence from the simplified ZX graph.
        :param zx_graph: The simplified ZX graph object
        :return: Extracted physical gate sequence, where each gate is represented as (gate_type, qubits, phase)
        """
        import qiskit.qasm2
        from qiskit import QuantumCircuit, qasm2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        # Convert the ZX graph to a quantum circuit using the from_graph method
        circuit = zx.Circuit.from_graph(zx_graph)

        # Convert the ZX circuit to a QASM string
        qasm_str = circuit.to_qasm()

        # Read the QASM string and generate a quantum circuit object using qiskit
        qc = QuantumCircuit.from_qasm_str(qasm_str)

        # Generate a pass manager with optimization level 3
        pm = generate_preset_pass_manager(optimization_level=3, basis_gates=["x", "y", "z", "h", "rx", "ry", "rz", "cz", "sx", "sxdg", "s", "sdg", "t", "tdg"])

        # Run the optimization and get the optimized circuit
        qc_trans = pm.run(qc)

        # Convert the optimized quantum circuit to a QASM string
        optimized_qasm_str = qasm2.dumps(qc_trans)

        # Remove unnecessary header information from the QASM file
        qasm_lines = optimized_qasm_str.splitlines()
        physical_gate_sequence = ""

        for line in qasm_lines:
            line = line.strip()
            if line.startswith("qreg") or line.startswith("include") or line.startswith("OPENQASM"):
                continue  # Skip these lines

            # Retain only the lines containing actual quantum gate operations
            if line:
                physical_gate_sequence += line + "\n"

        return physical_gate_sequence

        
    def calculate_gate_metrics_from_sequence(self, physical_gate_sequence: str):
        """
        Calculate the number of CNOT gates, T gates, and S gates in the physical gate sequence.

        :param physical_gate_sequence: Physical gate sequence string, where each gate is formatted as 'GATE QUBITS PARAM'.
        :return: (cnot_count, t_count, s_count) - the counts of the respective metrics
        """
        # Initialize counters
        cnot_count = 0
        t_count = 0
        s_count = 0

        # Iterate over each line in the physical gate sequence
        lines = physical_gate_sequence.splitlines()
        for line in lines:
            # Extract the gate type
            parts = line.split()
            gate_type = parts[0]

            # Count the occurrences of specific gates
            if gate_type == "CZ":  # CNOT gate (corresponding to CZ gate in QCIS instruction set)
                cnot_count += 1
            elif gate_type in {"T", "TD"}:  # T gate or T† gate
                t_count += 1
            elif gate_type in {"S", "SD", "H"}:  # S gate, S† gate, and Hadamard gate
                s_count += 1

        return cnot_count, t_count, s_count

    def calculate_circuit_depth(self, zx_graph):
        """
        Calculate the circuit depth of the physical gate sequence.
        :param zx_graph: The simplified ZX graph object
        :return: Circuit depth (int)
        """
        # Convert the ZX graph to a quantum circuit using the from_graph method
        circuit = zx.Circuit.from_graph(zx_graph)

        # Use the built-in depth method in PyZX to calculate circuit depth
        circuit_depth = circuit.depth()

        return circuit_depth

    def save_physical_gate_sequence_to_txt(self, file_path: str):
        """
        Save the physical gate sequence to a text file.
        :param file_path: Path to the output text file
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            for gate, qubits, param in self.physical_gate_sequence:
                qubit_str = ' '.join([f"Q{q}" for q in qubits])
                if param is not None:
                    file.write(f"{gate} {qubit_str} {param}\n")
                else:
                    file.write(f"{gate} {qubit_str}\n")
        print(f"Physical gate sequence saved to {file_path}")



