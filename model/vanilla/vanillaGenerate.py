import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

import numpy as np
from helper import *


# Parameters
num_graphs = 6400
num_nodes_per_graph = 10
num_classes = 2
batch_size = 128


# Generate random graphs
"""
graphs = []
for _ in range(num_graphs):
    class_label = torch.randint(0, num_classes, (1,))
    num_nodes = num_nodes_per_graph
    x = torch.randn(num_nodes, 3)  # Node features (3-dimensional)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))  # Random edge indices
    edge_attr = torch.randn(num_nodes * 2, 2)  # Random edge features (2-dimensional)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=class_label)
    graphs.append(data)


# Create a DataLoader
loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

# Iterate through the DataLoader
for batch in loader:
    print("Batch Node Features Shape:", batch.x.shape)
    print("Batch Edge Index Shape:", batch.edge_index.shape)
    print("Batch Edge Features Shape:", batch.edge_attr.shape)
    print("Batch Class Labels:", batch.y)
    print("--------------------------")
"""

# Cora dataset
"""
name_data = 'Cora'
dataset = Planetoid(root='../dataset/' + name_data, name=name_data)
dataset.transform = T.NormalizeFeatures()
"""

# MUTAG dataset
"""
path = "../dataset/"
name_data = 'MUTAG'
dataset = TUDataset(path, name=name_data).shuffle()

print(f"Number of classes in {name_data}:", dataset.num_classes)
print(f"Number of node features in {name_data}:", dataset.num_node_features)
"""

class QM9DataModule:
    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        target_idx: int = 5,
        seed: float = 420,
    ) -> None:
        """Encapsulates everything related to the dataset

        Parameters
        ----------
        train_ratio : float, optional
            fraction of data used for training, by default 0.8
        val_ratio : float, optional
            fraction of data used for validation, by default 0.1
        test_ratio : float, optional
            fraction of data used for testing, by default 0.1
        target_idx : int, optional
            index of the target (see torch geometric docs), by default 5 (electronic spatial extent)
            (https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html?highlight=qm9#torch_geometric.datasets.QM9)
        seed : float, optional
            random seed for data split, by default 420
        """
        assert sum([train_ratio, val_ratio, test_ratio]) == 1
        self.target_idx = target_idx
        self.num_examples = len(self.dataset())
        rng = np.random.default_rng(seed)
        self.shuffled_index = rng.permutation(self.num_examples)
        self.train_split = self.shuffled_index[: int(self.num_examples * train_ratio)]
        self.val_split = self.shuffled_index[
            int(self.num_examples * train_ratio) : int(
                self.num_examples * (train_ratio + val_ratio)
            )
        ]
        self.test_split = self.shuffled_index[
            int(self.num_examples * (train_ratio + val_ratio)) : self.num_examples
        ]

    def dataset(self, transform=None) -> QM9:
        dataset = QM9(
            root="../dataset/QM9",
            pre_filter=lambda data: num_heavy_atoms(data) < 9,
            pre_transform=add_complete_graph_edge_index,
        )
        dataset.data.y = dataset.data.y[:, self.target_idx].view(-1, 1)
        return dataset

    def loader(self, split, **loader_kwargs) -> DataLoader:
        dataset = self.dataset()[split]
        return DataLoader(dataset, **loader_kwargs)

    def train_loader(self, **loader_kwargs) -> DataLoader:
        return self.loader(self.train_split, shuffle=True, **loader_kwargs)

    def val_loader(self, **loader_kwargs) -> DataLoader:
        return self.loader(self.val_split, shuffle=False, **loader_kwargs)

    def test_loader(self, **loader_kwargs) -> DataLoader:
        return self.loader(self.test_split, shuffle=False, **loader_kwargs)

  
dataset = QM9(
    root="../dataset/QM9",
    # Filter out molecules with more than 8 heavy atoms
    pre_filter=lambda data: num_heavy_atoms(data) < 9,
    # implement point cloud adjacency as a complete graph
    pre_transform=add_complete_graph_edge_index,
)

print(f"Num. examples in QM9 restricted to molecules with at most 8 heavy atoms: {len(dataset)}")

data_module = QM9DataModule()
