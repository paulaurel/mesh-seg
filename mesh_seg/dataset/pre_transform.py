import torch

from torch_geometric.transforms import BaseTransform


class NormalizeUnitSphere(BaseTransform):
    """
    """

    @staticmethod
    def _re_center(pos):
        centroid = torch.mean(pos, dim=0)
        return pos - centroid

    @staticmethod
    def _re_scale_to_unit_length(pos):
        max_dist = torch.max(torch.norm(pos, dim=1))
        return pos / max_dist

    def __call__(self, data):
        if data.vertices is not None:
            data.vertices = self._re_scale_to_unit_length(
                self._re_center(data.vertices)
            )

        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)
