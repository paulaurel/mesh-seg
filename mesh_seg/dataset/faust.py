from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, extract_zip

from .io import load_mesh


class SegmentationFaust(InMemoryDataset):

    seg_classes = dict(
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

    def __init__(self, root, train: bool = True, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> str:
        return "MPI-FAUST.zip"

    @property
    def processed_file_names(self) -> list:
        return ["training.pt", "test.pt"]

    def download(self):
        raise RuntimeError(
            f"Dataset not found. Please download '{self.raw_file_names}'"
            f" and move it to '{self.raw_dir}'")

    def process(self):
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)
        path_to_meshes = Path(self.raw_dir) / 'MPI-FAUST' / 'training' / 'registrations'
        mesh_filenames = path_to_meshes.glob("*.ply")

        data_list = []
        for mesh_filename in sorted(mesh_filenames):
            vertices, faces = load_mesh(mesh_filename)
            data = Data(vertices=vertices, face=faces)
            # TODO: @paulaurel fill data.semantic_labels with semantic class labels
            data.semantic_labels = torch.randint(
                low=0,
                high=max(self.seg_classes.values()),
                size=(data.vertices.shape[0], 1),
            )
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list[:80]), self.processed_paths[0])
        torch.save(self.collate(data_list[80:]), self.processed_paths[1])
