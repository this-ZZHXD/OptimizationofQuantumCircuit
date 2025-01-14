import unittest
import networkx as nx
import pyzx as zx
from env_zx import QuantumCircuitSimplificationEnv as EnvZx

# 示例测试函数
class TestQuantumCircuitSimplificationEnv(unittest.TestCase):
    def setUp(self):
        # 创建测试用的chip_graph
        self.chip_graph = nx.Graph()
        self.chip_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        self.env = EnvZx(self.chip_graph)

    def test_is_graph_connected_after_removal_connected(self):
        # 测试移除顶点后图仍然连通的情况
        vertex_to_remove = 2
        self.assertTrue(self.env.is_graph_connected_after_removal(vertex_to_remove))

    def test_is_graph_connected_after_removal_disconnected(self):
        # 测试移除顶点后图不再连通的情况
        self.chip_graph.remove_edge(1, 2)  # 使图中部分边断开
        vertex_to_remove = 2
        self.assertFalse(self.env.is_graph_connected_after_removal(vertex_to_remove))

    def test_is_graph_connected_after_removal_boundary(self):
        # 测试移除边界顶点的情况
        self.chip_graph.add_node(5)  # 增加一个孤立的边界顶点
        vertex_to_remove = 5
        self.assertFalse(self.env.is_graph_connected_after_removal(vertex_to_remove))

    def test_is_graph_connected_after_removal_full_connected(self):
        # 测试完全连通图，移除任意顶点后图仍然连通
        full_graph = nx.complete_graph(5)
        self.env = EnvZx(full_graph)
        vertex_to_remove = 3
        self.assertTrue(self.env.is_graph_connected_after_removal(vertex_to_remove))

    def test_is_graph_connected_after_removal_cycle(self):
        # 测试循环图，移除任意顶点后图仍然连通
        cycle_graph = nx.cycle_graph(5)
        self.env = EnvZx(cycle_graph)
        vertex_to_remove = 2
        self.assertTrue(self.env.is_graph_connected_after_removal(vertex_to_remove))

    def test_is_graph_connected_after_removal_tree(self):
        # 测试树结构，移除叶子节点后图仍然连通
        tree_graph = nx.balanced_tree(2, 3)  # 生成一个2-叉3层的树
        self.env = EnvZx(tree_graph)
        leaf_vertex = 6  # 树的叶子节点之一
        self.assertTrue(self.env.is_graph_connected_after_removal(leaf_vertex))

if __name__ == "__main__":
    unittest.main()
