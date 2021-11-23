from io import BytesIO
from multiprocessing import Pool

import torch
from joblib import Memory, cpu_count

import PIL
import numpy as np
import potpourri3d as pp3d
import matplotlib.pyplot as plt

CACHE_DIR = '/content/sample_data'
CACHE_MEMORY = Memory(CACHE_DIR, verbose=0)

GEO_SOLVER = None


def _compute_geodesics(source_idx):
    return GEO_SOLVER.compute_distance(source_idx)


def compute_geodesic_matrix(points, faces):
    global GEO_SOLVER

    if faces.shape[0] == 3:
        faces = np.transpose(faces)

    GEO_SOLVER = pp3d.MeshHeatMethodDistanceSolver(points, faces)
    num_vertices = points.shape[0]

    pool = Pool(processes=cpu_count())
    all_geodesics = [
        pool.apply(_compute_geodesics, args=(source_idx,))
        for source_idx in range(num_vertices)
    ]
    return np.vstack(all_geodesics)


cached_compute_geodesic_matrix = CACHE_MEMORY.cache(compute_geodesic_matrix)


def _count_assignments_within_threshold(geodesic_dists, threshold):
    num_dists = len(geodesic_dists)
    num_dists_within_threshold = np.sum(geodesic_dists <= threshold)
    return num_dists_within_threshold / num_dists


def compute_geodesic_auc(assignment_accuracy, threshold_step):
    return np.sum(assignment_accuracy[:-1] * threshold_step)


def compute_assignment_error(geodesic_matrix, pred_idx, gt_idx, threshold_step=0.01):
    geodesic_dists = geodesic_matrix[pred_idx, gt_idx] / np.max(geodesic_matrix)
    thresholds = np.arange(threshold_step, 1.0 + threshold_step, threshold_step)

    pool = Pool(processes=cpu_count())
    assignment_error = np.array(
        [pool.apply(_count_assignments_within_threshold, args=(geodesic_dists, threshold))
         for threshold in thresholds
         ]
    )
    assignment_auc = compute_geodesic_auc(assignment_error, threshold_step)
    return assignment_error, assignment_auc


def evaluate_assignment_error(points, faces, pred_idx):
    geodesic_matrix = cached_compute_geodesic_matrix(points, faces)
    gt_idx = np.arange(points.shape[0])
    return compute_assignment_error(geodesic_matrix, pred_idx, gt_idx)


def plot_assignment_error(assignment_accuracy, assignment_auc):
    plt.figure(figsize=(8, 6))
    plt.plot(
        np.linspace(0.0, 1.0, len(assignment_accuracy), endpoint=True),
        assignment_accuracy,
    )
    plt.ylabel("Correspondence Accuracy (%)")
    plt.xlabel("Geodesic Error")
    plt.title(
        f"Correspondence Accuracy vs. Geodesic Error"
        f" (AUC: {assignment_auc:.2f})"
    )
    buffer = BytesIO()
    plt.savefig(buffer, format="jpeg")
    buffer.seek(0)
    return np.asarray(PIL.Image.open(buffer))


def compute_assignment_accuracy(pred_assignment):
    gt_assignment = torch.arange(pred_assignment.shape[1], device=pred_assignment.device)
    return torch.mean(pred_assignment.squeeze() == gt_assignment)


def compute_class_label_accuracy(predictions, gt_seg_labels):
    predicted_class_labels = predictions.argmax(dim=-1, keepdim=True)
    if predicted_class_labels.shape != gt_seg_labels.shape:
        raise ValueError("Expected Shapes to be equivalent")
    num_correct_class_label_assignments = (predicted_class_labels == gt_seg_labels).sum()
    tot_num_class_label_assignments = predicted_class_labels.shape[0]
    return num_correct_class_label_assignments / tot_num_class_label_assignments