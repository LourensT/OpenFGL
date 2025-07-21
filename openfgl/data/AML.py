from torch_geometric.data import InMemoryDataset, Data
from typing import Optional, Callable, List
import torch
import os.path as osp


class AMLDataset(InMemoryDataset):

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    # TODO move the logic here into self.process, and then do not overwrite the load method
    def load(self, path: str) -> None:
        print(f"Loading dataset from {path}")

        hetdata = torch.load(path, weights_only=False)['test']

        self.data = Data( 
            x=hetdata['node']['x'], 
            edge_index=hetdata['node','to','node']['edge_index'], 
            edge_attr=hetdata['node','to','node']['edge_attr'],
            y = hetdata['node', 'to', 'node']['y']
        )

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        raise NotImplementedError

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self) -> None:
        print("Processing dataset...")
        # TODO actually process the dataset instead of using the cached data from
        pass

    def __repr__(self) -> str:
        return f'{self.name}()'



if __name__ == "__main__":
    data = AMLDataset(root="/mnt/lourens/data/AML", name="AML-Small-High-LI")
    print(data[0])

    # print(data.x.shape, data.edge_index.shape, data.y.shape)
    # print(data.num_nodes, data.num_edges, data.num_features, data.num_classes)
    # print(data.train_mask.shape, data.val_mask.shape, data.test_mask.shape)
    # print(data.train_mask.sum().item(), data.val_mask.sum().item(), data.test_mask.sum().item())