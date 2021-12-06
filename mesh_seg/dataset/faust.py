from pathlib import Path
from functools import lru_cache

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from .io import load_mesh


class SegmentationFaust(InMemoryDataset):
    """A segmented version of the MPI FAUST humanoid mesh dataset."""
    map_seg_label_to_id = dict(
        head=0,
        torso=1,
        left_arm=2,
        left_hand=3,
        right_arm=4,
        right_hand=5,
        left_upper_leg=6,
        left_lower_leg=7,
        left_foot=8,
        right_upper_leg=9,
        right_lower_leg=10,
        right_foot=11,
    )

    def __init__(self, root, train: bool = True, pre_transform=None):
        """
        Parameters
        ----------
        root: PathLike
            Root directory where the dataset should be saved.
        train: bool
            Whether to load training data or test data.
        pre_transform: Optional[Callable]
            A function that takes in a torch_geometric.data.Data object
            and outputs a transformed version. Note that the transformed
            data object will be saved to disk.

        """
        super().__init__(root, pre_transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def processed_file_names(self) -> list:
        return ["training.pt", "test.pt"]

    @property
    @lru_cache(maxsize=32)
    def _segmentation_labels(self):
        """Extract segmentation labels."""
        path_to_seg_labels = Path(self.root) / "semantic_labels" / "segmentations.npz"
        seg_labels = np.load(str(path_to_seg_labels))["segmentation_labels"]
        return torch.from_numpy(seg_labels).type(torch.int64)

    def _mesh_filenames(self):
        """Extract all mesh filenames."""
        path_to_meshes = Path(self.root) / "training" / "registrations"
        return path_to_meshes.glob("*.ply")

    def process(self):
        """Process the raw meshes files and their corresponding segmentation labels into Data objects."""
        data_list = []
        for mesh_filename in sorted(self._mesh_filenames()):
            vertices, faces = load_mesh(mesh_filename)
            data = Data(x=vertices, face=faces)
            data.segmentation_labels = self._segmentation_labels
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list[:80]), self.processed_paths[0])
        torch.save(self.collate(data_list[80:]), self.processed_paths[1])
