from pathlib import Path

import torch
import trimesh


def load_mesh(mesh_filename: Path):
    """Extract vertices and faces from raw mesh file.

    Parameters
    ----------
    mesh_filename: PathLike
        Path to mesh `.ply` file.

    Returns
    -------
    vertices: torch.tensor
        Float tensor of size (|V|, 3), where each row
        specifies the spatial position of a vertex in 3D space.
    faces: torch.tensor
        Intger tensor of size (|M|, 3), where each row
        defines a traingular face.
    """
    mesh = trimesh.load_mesh(mesh_filename, process=False)
    vertices = torch.from_numpy(mesh.vertices).to(torch.float)
    faces = torch.from_numpy(mesh.faces)
    faces = faces.t().to(torch.long).contiguous()
    return vertices, faces
