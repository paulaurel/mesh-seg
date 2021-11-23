import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.transforms import FaceToEdge, Compose

from metrics import (
    evaluate_assignment_error,
    plot_assignment_error,
    compute_assignment_accuracy,
    compute_class_label_accuracy,
)
from utils import groupwise
from models.network import MeshCorrSeg
from dataset.pre_transform import NormalizeUnitSphere
from dataset.faust import SegmentationFaust, SEGMENTATION_COLORS


def training_loop(net, train_data_loader: DataLoader, optimizer, multitask_loss_fn, loss_weights, device):
    net.training_loop()
    cumulative_loss = 0.0
    for data_s, data_t in groupwise(train_data_loader):
        data_s, data_t = [data.to(device) for data in (data_s, data_t)]
        optimizer.zero_grad()
        output = net(data_s, data_t)
        loss = multitask_loss_fn(*output, data_s, data_t, loss_weights)
        loss.backward()
        cumulative_loss += loss.item()
        optimizer.step()
    return cumulative_loss / len(train_data_loader)


def multitask_loss(assignment_matrix, pred_class_labels_s, pred_class_labels_t, data_s, data_t, weighting):
    def _segmentation_loss():
        seg_loss_s = F.cross_entropy(
            input=pred_class_labels_s,
            target=data_s.segmentation_labels.squeeze(),
        )
        seg_loss_t = F.cross_entropy(
            input=pred_class_labels_t,
            target=data_t.segmentation_labels.squeeze(),
        )
        return seg_loss_s + seg_loss_t

    def _correspondence_loss():
        gt_assignment = torch.arange(assignment_matrix.shape[0], device=assignment_matrix.device)
        return F.cross_entropy(assignment_matrix, gt_assignment)

    return sum(loss * weight for loss, weight in zip((_segmentation_loss(), _correspondence_loss()), weighting))


def _class_predictions_to_summary(data, predicted_class_labels, writer, map_seg_id_to_color, epoch, label_idx):
    def _map_class_label_to_color(seg_ids, map_seg_id_to_color):
        return torch.vstack(
            [map_seg_id_to_color[int(seg_ids[idx])] for idx in range(seg_ids.shape[0])]
        )

    mesh_colors = _map_class_label_to_color(predicted_class_labels, map_seg_id_to_color)
    writer.add_mesh(
        f"segmentation/test_{label_idx:02d}",
        vertices=data.x.unsqueeze(0),
        colors=mesh_colors.unsqueeze(0),
        faces=data.face.t().unsqueeze(0),
        global_step=epoch,
    )


@torch.no_grad()
def visualize_class_predictions(net, data_s, data_t, device, writer, map_seg_id_to_color, epoch):
    def _predicted_class_labels(soft_predictions):
        return soft_predictions.argmax(dim=-1, keepdim=True)

    _, soft_class_labels_s, soft_class_labels_t = net(data_s, data_t)
    data_s, data_t = [data.to(device) for data in (data_s, data_t)]
    _class_predictions_to_summary(
        data=data_s,
        predicted_class_labels=_predicted_class_labels(soft_class_labels_s),
        writer=writer,
        map_seg_id_to_color=map_seg_id_to_color,
        epoch=epoch,
        label_idx=0,
    )
    _class_predictions_to_summary(
        data=data_s,
        predicted_class_labels=_predicted_class_labels(soft_class_labels_s),
        writer=writer,
        map_seg_id_to_color=map_seg_id_to_color,
        epoch=epoch,
        label_idx=1,
    )


@torch.no_grad()
def evaluate_metrics(data_loader, net, device, writer, epoch, dataset_label):
    class_accuracies, assignment_accuracies, assignment_errors, assignment_aucs = [], [], [], []
    for data_s, data_t in groupwise(data_loader):
        data_s, data_t = [data.to(device) for data in (data_s, data_t)]
        assignment_matrix, pred_class_labels_s, pred_class_labels_t = net(data_s, data_t)

        class_accuracies.append(compute_class_label_accuracy(pred_class_labels_s, data_s.segmentation_labels))
        class_accuracies.append(compute_class_label_accuracy(pred_class_labels_t, data_t.segmentation_labels))

        pred_assignment = assignment_matrix.max(1).indices.squeeze()
        assignment_accuracies.append(compute_assignment_accuracy(pred_assignment))
        assignment_error, assignment_auc = evaluate_assignment_error(
            points=data_s.x.numpy(),
            faces=data_s.face.numpy(),
            pred_idx=pred_assignment.numpy(),
        )
        assignment_errors.append(assignment_error)
        assignment_aucs.append(assignment_auc)

    writer.add_scalar(f"seg_class_accuracy/{dataset_label}", np.mean(class_accuracies), epoch)
    writer.add_scalar(f"assignment_accuracy/{dataset_label}", np.mean(assignment_accuracies), epoch)

    assignment_error_plot = plot_assignment_error(np.mean(assignment_errors, axis=1), np.mean(assignment_aucs))
    writer.add_image(f"assignment_error/{dataset_label}", assignment_error_plot)


@torch.no_grad()
def perform_evaluation(net, train_loader, test_loader, device, writer, epoch):
    net.eval()
    evaluate_metrics(train_loader, net, device, writer, epoch, dataset_label="train")
    evaluate_metrics(test_loader, net, device, writer, epoch, dataset_label="test")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = MeshCorrSeg(
        in_features=3,
        num_seg_classes=12,
        encoder_channels=[8, 16],
        conv_channels=[32, 64, 128, 64],
        class_decoder_channels=[32, 12],
        assignment_decoder_channels=[64, 128],
        num_heads=8,
        sinkhorn_iterations=5,
    ).to(device)

    pre_transform = Compose([FaceToEdge(remove_faces=False), NormalizeUnitSphere()])
    root = "/home/diepaul/cs224-project/MPI-FAUST"
    train_data = SegmentationFaust(
        root,
        train=True,
        pre_transform=pre_transform,
    )
    test_data = SegmentationFaust(
        root,
        train=False,
        pre_transform=pre_transform,
    )

    map_seg_id_to_color = dict(
        (_value, SEGMENTATION_COLORS[_key])
        for _key, _value in train_data.map_seg_label_to_id.items()
    )

    train_loader = DataLoader(train_data,  shuffle=True)
    test_loader = DataLoader(test_data, shuffle=False)

    lr = 0.001
    num_epochs = 20000

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    writer = SummaryWriter(log_dir="/home/diepaul/cs224-project/logs")
    loss_weights = [0.5, 0.0]
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")

        if epoch == 1000:
            loss_weights = [0.5, 0.2]

        train_loss = training_loop(net, train_loader, optimizer, multitask_loss, device, loss_weights)
        writer.add_scalar('mean-ce-loss/train', train_loss, epoch)

        if epoch % 50 == 0:
            perform_evaluation(net, train_loader, test_loader, device, writer, epoch)
            visualize_class_predictions(net, test_data[0], test_data[1], device, writer, map_seg_id_to_color, epoch)


if __name__ == "__main__":
    main()
