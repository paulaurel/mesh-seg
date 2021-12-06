import torch

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class NormalizeUnitSphere(BaseTransform):
    """Center and normalize node-level features to unit length."""

    @staticmethod
    def _re_center(x):
        """Recenter node-level features onto feature centroid."""
        centroid = torch.mean(x, dim=0)
        return x - centroid

    @staticmethod
    def _re_scale_to_unit_length(x):
        """Rescale node-level features to unit-length."""
        max_dist = torch.max(torch.norm(x, dim=1))
        return x / max_dist

    def __call__(self, data: Data):
        if data.x is not None:
            data.x = self._re_scale_to_unit_length(self._re_center(data.x))

        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)
