from pathlib import Path

import torch
import trimesh


def load_mesh(mesh_filename: Path):
    mesh = trimesh.load_mesh(mesh_filename, process=False)
    vertices = torch.from_numpy(mesh.vertices).to(torch.float)
    faces = torch.from_numpy(mesh.faces)
    faces = faces.t().to(torch.long).contiguous()
    return vertices, faces
