## Project Structure

### Directory Structure
```  
├── data  
│   ├── log               # Model training logs  
│   ├── model_checkpoints # Model checkpoint files  
│   ├── output            # Output results  
│   ├── Dataset           # Kaggle dataset  
├── env  
│   ├── env_zx.py         # Reinforcement learning environment file  
│   ├── circuit_utils.py  # Utility functions file  
├── RL  
│   ├── PPO.py            # PPO algorithm implementation  
│   ├── replaybuffer.py   # Replay Buffer implementation  
├── main.py               # Main program entry point
```

## Quick Start

### Data Preparation
Store quantum circuit data files in the `data` directory.  
- Kaggle dataset: `data/archive` directory  

### Data Switching (Changing paths for different datasets)
Modify the `__init__` function in the environment:
```python
self.input_folder = os.path.abspath(os.path.join(self.base_dir, '..', 'data', 'simplified_data'))
```

### Running the Main Program
Start reinforcement learning training using the following command:
```
python main.py
```

# Data Processing

The data processing workflow in this project revolves around quantum circuit simplification tasks, aiming to convert the gate sequence or graph structure of quantum circuits into input data manageable by the agent in the reinforcement learning environment. Below is the detailed data processing workflow:

---

## **1. Data Loading**

### **Supported File Formats**
The project supports the following quantum circuit description formats:
1. **QASM Files**:  
   - Standard format containing quantum gate sequences, which can be directly loaded.  
   - Use `zx.Circuit.load()` to parse QASM files and convert them into ZX graph objects.  

2. **TXT Files**:  
   - Plain text descriptions of quantum gates.  
   - Load gate sequences using `load_gate_sequence_from_txt`, convert them into QASM strings, and then into ZX graphs.  

3. **QCIS Files**:  
   - Similar to TXT files, representing quantum gate sequences.  
   - Load using `load_gate_sequence_from_qcis` and convert to QASM format.  

### **QASM Conversion**
Regardless of the file format, all data is ultimately converted into ZX graph objects. The specific process is as follows:
```python
self.circuit = zx.Circuit.load(file_path)  # Load QASM file
self.zx_graph = self.circuit.to_graph()    # Convert to ZX graph object
```

# Data Processing

## **2. Node Feature Processing**

### **Node Feature Design**

Each node feature vector has a length of 14 and includes the following information:

1. **Node Type**:  
   - Includes \( Z \) type, \( X \) type, and boundary nodes.  
   - Extracted using `get_node_type` and represented through one-hot encoding.  

2. **Node Phase**:  
   - Represents the angle of the quantum gate operation.  
   - Extracted using `get_node_phase` and normalized to the range \([-\pi, \pi]\).  

3. **Quantum Gate Type**:  
   - Includes \( T \), \( S \), \( H \), \( RX \), \( RY \), \( RZ \), etc.  
   - Extracted using `get_gate_type` and represented through one-hot encoding.  

### **Node Feature Generation**

Code example for generating the node feature matrix:
```python
node_features = np.zeros((total_nodes, 14))  # Initialize node feature matrix
for i, node in enumerate(self.zx_graph.vertices()):
    node_type = self.get_node_type(node)
    phase = self.get_node_phase(node)
    gate_type = self.get_gate_type(node)
    node_features[i][:3] = self.one_hot_encode(node_type, 3)  # Node type
    node_features[i][3] = phase  # Phase
    node_features[i][4:10] = self.one_hot_encode(gate_type, 6)  # Quantum gate type
```

## **3. Edge Feature Processing**

### **Edge Feature Design**

Each edge feature vector has a length of 3 and includes the following information:

1. **Edge Type**:  
   - Includes normal edges and Hadamard edges.  
   - Extracted using `get_edge_type` and represented through one-hot encoding.  

2. **Edge Weight**:  
   - Determined based on the type of quantum gate operation.  
   - Examples:  
     - **CZ Gate**: Weight of 5.  
     - **Rotation Gates (RX, RY, RZ)**: Weight of 2.  
     - **Hadamard Gate (H)**: Weight of 3.  

### **Edge Feature Generation**

Code example for generating the edge feature matrix:
```python
edge_attributes = np.zeros((num_edges, 3))  # Initialize edge feature matrix
for i, edge in enumerate(self.zx_graph.edges()):
    edge_type = self.get_edge_type(edge)
    edge_attributes[i][:2] = self.one_hot_encode(edge_type, 2)  # Edge type
    edge_attributes[i][2] = self.get_edge_weight(edge)  # Edge weight
```

### **Generation Description**

1. **Edge Type (One-Hot Encoding)**:  
   - Normal edges are represented as `[1, 0]`.  
   - Hadamard edges are represented as `[0, 1]`.  

2. **Edge Weight**:  
   - Extracted using the `get_edge_weight` function and filled into the third dimension of the edge feature vector.  

---

### **Edge Feature Example**

Assuming two edges:
- **Edge 1**: Normal edge with a weight of 5.  
- **Edge 2**: Hadamard edge with a weight of 3.  

The corresponding edge feature matrix:
```plaintext
edge_attributes = [
    [1, 0, 5],  # Features of Edge 1
    [0, 1, 3]   # Features of Edge 2
]
```
This matrix is later encapsulated as a `torch.Tensor` and used as part of the input to the reinforcement learning environment's state space.  

## **4. Edge Index Processing**

### **Definition**
The edge index matrix describes the connection relationships in the graph, with each edge represented by the indices of its source and target nodes.  

### **Edge Index Generation**

Code example for generating the edge index matrix:
```python
edge_indices = np.array([list(edge) for edge in self.zx_graph.edges()]).T  # (2, c)
```
- **Result**: The generated edge index matrix has a shape of (2, c), with each column representing the source and target node indices of an edge.  
- **Purpose**: Describes the connections between nodes in the ZX graph for processing by graph neural networks.  

## **5. Action Identifier Processing**

### **Definition**
Each node has an action identifier to distinguish real nodes from action nodes:
- **Real Nodes**: Identifier is -1, indicating they are non-operational.  
- **Action Nodes**: Identifier is a positive integer, indicating they can perform certain operations.  

---

### **Action Identifier Generation**

Code example for generating the action identifier tensor:
```python
y_values = np.full(num_real_nodes, -1, dtype=int)  # Initialize with -1
action_identifiers = np.arange(1, num_action_nodes + 1)
y_values = np.concatenate([y_values, action_identifiers])
```
- **Result**:  
  - The generated tensor contains identifiers for all nodes:  
    - Real nodes have an identifier of -1.  
    - Action nodes have identifiers starting from 1, incrementing sequentially.  
- **Purpose**: Used by the agent to identify which nodes are operational.  

## **6. Data Integration**

### **Definition**
Node features, edge features, edge indices, and action identifiers are integrated into input data for the reinforcement learning environment, encapsulated as a `torch_geometric.data.Data` object.  

---

### **Integration Code Example**
```python
data = Data(
    x=torch.tensor(node_features, dtype=torch.float32),        # Node features (a, 14)
    edge_index=torch.tensor(edge_indices, dtype=torch.long),  # Edge indices (2, c)
    edge_attr=torch.tensor(edge_attributes, dtype=torch.float32),  # Edge features (c, 3)
    identifiers=torch.tensor(y_values, dtype=torch.long)      # Action identifiers (a,)
)
```

### **Parameter Explanation**

- **`x`**:  
  - Node feature matrix with a shape of \((a, 14)\).  
  - Each row represents the features of a node, including node type, phase, and quantum gate type.  

- **`edge_index`**:  
  - Edge index matrix with a shape of \((2, c)\).  
  - Each column represents the source and target nodes of an edge.  

- **`edge_attr`**:  
  - Edge feature matrix with a shape of \((c, 3)\).  
  - Each row represents the features of an edge, including edge type and weight.  

- **`identifiers`**:  
  - Action identifier tensor with a shape of \((a,)\).  
  - Indicates which nodes are operational, with real nodes having an identifier of -1 and action nodes having positive integers.
