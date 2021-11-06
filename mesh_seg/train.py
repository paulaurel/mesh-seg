import torch
from torch_geometric.data import DataLoader
from torch_geometric.transforms import FaceToEdge, Compose
from dataset.faust import SegmentationFaust
from dataset.pre_transform import NormalizeUnitSphere
from models.network import MeshSeg

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

def accuracy(dataset, net, device):
    mean_accuracy=0
    for data in dataset:
        data = data.to(device)
        predictions = net(data)
        predicted_seg_labels = predictions.argmax(dim=-1, keepdim=True) 
        if predicted_seg_labels.shape != data.segmentation_labels.shape:
            raise ValueError("Expected Shapes to be equivalent")
        correct_assignments = (predicted_seg_labels == data.segmentation_labels).sum()
        num_assignemnts = predicted_seg_labels.shape[0]
        mean_accuracy += correct_assignments / num_assignemnts
    return mean_accuracy/len(dataset)

@torch.no_grad()
def test(net, train_data, test_data, device):
    net.eval()
    train_acc = accuracy(train_data, net, device)
    test_acc = accuracy(test_data, net, device)
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

    train_loader = DataLoader(train_data,  shuffle=True)
    test_loader = DataLoader(test_data, shuffle=False)

    lr = 0.01
    nb_epochs = 10

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, nb_epochs):
        loss = train(net, train_loader, optimizer, loss_fn, device)
        train_acc, test_acc = test(net, train_data, test_data, device)

