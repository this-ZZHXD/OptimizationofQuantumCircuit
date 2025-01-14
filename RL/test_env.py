
import torch
from torch_geometric.data import Data


class TestEnv:
    def __init__(self):
        self.data = self._new_graph()

    def _new_graph(self):
        x = torch.randn((114, 16))  # 10个节点，每个节点16个特征
        #print(x)
        edge_index = torch.tensor([[0, 1, 2, 3, 5, 4],
                                   [1, 2, 3, 4, 0, 5]])  # 边索引
        #print(edge_index)
        edge_attr = torch.randn((6, 6))  # 5条边，每条边6个特征
        # print(edge_attr)
        identifiers = torch.randperm(100)[:10]
        for i in torch.randint(0, 10, (2,)):
            identifiers[i] = -1
        print(identifiers)
        self.data = Data(x=x,  # 节点特征 (10 个节点，每个有 16 个特征)
                         edge_index=edge_index,  # 边的索引
                         edge_attr=edge_attr,  # 边的特征 (20 条边，每条边有 6 个特征)
                         identifiers=identifiers)  # 节点标签
        print(self.data)

        return self.data

    def step(self, action):
        node = torch.where(self.data.identifiers == action)[0]
        reward = torch.sum(self.data.x[node])
        self.data.x[node] += 1
        done = True if torch.max(self.data.x[node]) > 100 else False
        return self.data, reward, done, "", ""

    def reset(self):
        self._new_graph()
        return self.data, ''


if __name__ == '__main__':
    env = TestEnv()
    for i in range(1):
        env.reset()

