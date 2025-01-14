import json
import networkx as nx
from typing import List, Tuple, Union
import pyzx as zx
import os
import numpy as np
from typing_extensions import TypeAlias
from sympy import sympify, SympifyError
import math
import torch
import random

GateSeq: TypeAlias = List[Tuple[str, Tuple[str, ...], Union[float, None]]]  # A sequence of quantum gates operations

def load_gate_sequence_from_txt(file_path: str) -> List[Tuple[str, List[str], Union[float, int, None]]]:
    """
    Load quantum gate sequences from a TXT file.

    Parameters:
    - file_path: Path to the file

    Returns:
    - gate_sequence: A list containing (gate, qubits, parameter)
    """
    gate_sequence = []
    valid_gates = {"X", "Y", "Z", "H", "RX", "RY", "RZ", "X2P", "X2M", "Y2P", "Y2M", "S", "SD", "T", "TD", "CZ", "M"}

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) < 2 or parts[0] not in valid_gates:
                continue  # Ignore invalid lines

            gate = parts[0]
            qubits = parts[1:-1] if len(parts) > 2 and parts[-1].replace('.', '', 1).replace('-', '', 1).isdigit() else parts[1:]
            param = None

            # Check if the last element is a numeric value (float or int)
            if len(parts) > 2 and parts[-1].replace('.', '', 1).replace('-', '', 1).isdigit():
                try:
                    if '.' in parts[-1]:
                        param = float(parts[-1])  # Parse as float if it contains a decimal point
                    else:
                        param = int(parts[-1])  # Parse as integer otherwise
                except ValueError:
                    print(f"[WARNING] Invalid parameter: {parts[-1]}. Skipping.")
                    param = None

            # Ensure qubits are parsed into a list of individual quantum bits
            qubits = [q.strip() for q in qubits]

            # Add the parsed gate, qubits, and parameter to the sequence
            gate_sequence.append((gate, qubits, param))

    return gate_sequence


def load_gate_sequence_from_qcis(file_path: str) -> List[Tuple[str, List[str], Union[float, None]]]:
    """
    Load quantum gate sequences from a .qcis file.
    """
    gate_sequence = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            gate = parts[0]
            qubits = parts[1:2]
            param = None

            if len(parts) > 2:  # Case where a parameter exists
                param_str = parts[2]
                try:
                    # Directly parse the value without handling symbolic parameters
                    param = float(param_str)
                except ValueError:
                    print(f"Error parsing parameter {param_str} as float.")
                    param = None

            gate_sequence.append((gate, qubits, param))

    print(f"[DEBUG] Loaded gate sequence from {file_path}: {gate_sequence}")
    return gate_sequence


def gate_sequence_to_tensor(gate_sequence, qubit_count_int):
    """
    Convert a gate sequence into a tensor representation.

    Parameters:
    - gate_sequence: List of quantum gates
    - qubit_count_int: Total number of qubits

    Returns:
    - tensor: A tensor representing the quantum gate sequence
    """
    # Create a mapping to ensure each floating-point index is uniquely mapped to an integer index
    qubit_mapping = {}
    max_index = 0  # Track the maximum integer index
    tensor = torch.zeros((1, qubit_count_int, 2, len(gate_sequence)), dtype=torch.float32)  # Set height to 2

    for i, (gate, qubits, param) in enumerate(gate_sequence):
        for qubit in qubits:
            float_index = float(qubit[1:])  # Extract the floating-point index
            if float_index not in qubit_mapping:
                # Assign a unique integer index for each floating-point index
                qubit_mapping[float_index] = len(qubit_mapping)
            qubit_index = qubit_mapping[float_index]  # Retrieve the integer index

            # Check and adjust the tensor dimensions
            if qubit_index >= tensor.size(1):
                # Extend the tensor dimensions instead of reinitializing
                new_tensor = torch.zeros((1, qubit_index + 1, 2, len(gate_sequence)), dtype=torch.float32)
                new_tensor[:, :tensor.size(1), :, :] = tensor  # Retain existing data
                tensor = new_tensor  # Update the tensor to accommodate the new qubit_index

            # Update the tensor values
            tensor[0, qubit_index, :, i] = 1.0  # Fill tensor values for the gate at the corresponding qubit index

    return tensor


