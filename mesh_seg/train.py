import torch
from torch_geometric.data import DataLoader
from torch_geometric.transforms import FaceToEdge, Compose
from dataset.faust import SegmentationFaust
from dataset.pre_transform import NormalizeUnitSphere
from models.network import MeshSeg
from torch.utils.tensorboard import SummaryWriter


SEGMENTATION_COLORS = dict(
    head=torch.tensor([255, 0, 0], dtype=torch.int),
    torso=torch.tensor([255, 0, 255], dtype=torch.int),
    left_arm=torch.tensor([255, 255, 0], dtype=torch.int),
    left_hand=torch.tensor([255, 128, 0], dtype=torch.int), 
    right_arm=torch.tensor([0, 255, 0], dtype=torch.int),
    right_hand=torch.tensor([0, 255, 128], dtype=torch.int),
    left_upper_leg=torch.tensor([0, 128, 255], dtype=torch.int),
    left_lower_leg=torch.tensor([0, 255, 255], dtype=torch.int),
    left_foot=torch.tensor([0, 0, 255], dtype=torch.int),
    right_upper_leg=torch.tensor([128, 0, 255], dtype=torch.int),
    right_lower_leg=torch.tensor([128, 255, 0], dtype=torch.int),
    right_foot=torch.tensor([255, 0, 128], dtype=torch.int)
)


def train(net, train_data, optimizer, loss_fn, device):
    net.train()
    cumulative_loss = 0.0
    for data in train_data:
        data = data.to(device)
        optimizer.zero_grad()
        out = net(data)
        loss = loss_fn(out, data.segmentation_labels.squeeze())
        loss.backward()
        cumulative_loss += loss.item()
        optimizer.step()
    return cumulative_loss / len(train_data)


def accuracy(predictions, gt_seg_labels): 
    predicted_seg_labels = predictions.argmax(dim=-1, keepdim=True)
    if predicted_seg_labels.shape != gt_seg_labels.shape:
        raise ValueError("Expected Shapes to be equivalent")
    correct_assignments = (predicted_seg_labels == gt_seg_labels).sum()
    num_assignemnts = predicted_seg_labels.shape[0]
    return correct_assignments / num_assignemnts


@torch.no_grad()
def visualize_predictions(net, data, device, writer, map_seg_id_to_color):
    def _map_seg_label_to_color(seg_ids, map_seg_id_to_color):
        return torch.vstack(
            [map_seg_id_to_color[int(seg_ids[idx])] for idx in range(seg_ids.shape[0])]
        )

    data = data.to(device)
    predictions = net(data)
    predicted_seg_labels = predictions.argmax(dim=-1, keepdim=True)
    mesh_colors = _map_seg_label_to_color(predicted_seg_labels, map_seg_id_to_color)
    writer.add_mesh(
        "segmentation/test",
        vertices=data.x,
        colors=mesh_colors,
        faces=data.faces.t()
    )


def evaluate_network(dataset, net, device):
    mean_accuracy = 0
    for data in dataset:
        data = data.to(device)
        predictions = net(data)
        mean_accuracy += accuracy(predictions, data.segmentation_labels)
    return mean_accuracy / len(dataset)


@torch.no_grad()
def test(net, train_data, test_data, device):
    net.eval()
    train_acc = evaluate_network(train_data, net, device)
    test_acc = evaluate_network(test_data, net, device)
    return train_acc, test_acc


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = MeshSeg(
        in_features=3,
        encoder_features=16,
        num_classes=12,
        conv_channels=[32, 64, 128, 64],
        num_heads=8,
    ).to(device)

    pre_transform = Compose([FaceToEdge(remove_faces=False), NormalizeUnitSphere()])
    root = "" # to do
    train_data = SegmentationFaust(
        root,
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

    lr = 0.01
    num_epochs = 10

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter()

    for epoch in range(num_epochs):
        train_loss = train(net, train_loader, optimizer, loss_fn, device)
        train_acc, test_acc = test(net, train_loader, test_loader, device)
        visualize_predictions(net, next(test_data), device, writer, map_seg_id_to_color)
        writer.add_scalar('mean-ce-loss/train', train_loss, epoch)
        writer.add_scalar('accuracy/train', train_acc, epoch)
        writer.add_scalar('accuracy/test', test_acc, epoch)

