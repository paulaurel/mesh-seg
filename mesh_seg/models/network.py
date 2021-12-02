import torch
import torch.nn as nn
import torch.nn.functional as F

from mesh_seg.utils import pairwise
from .sinkhorn import solve_log_optimal_transport
from .graph_convolutions.feat_steer_conv import FeatureSteeredConvolution


def get_conv_layers(channels: list, conv, conv_params):
    """Define basic multilayered graph convolution network architecture."""
    conv_layers = [
        conv(in_ch, out_ch, **conv_params) for in_ch, out_ch in pairwise(channels)
    ]
    return conv_layers


def get_mlp_layers(channels: list, activation, output_activation=nn.Identity):
    """Define basic multilayered perceptron network architecture."""
    layers = []
    *intermediate_layer_definitions, final_layer_definition = pairwise(channels)

    for in_ch, out_ch in intermediate_layer_definitions:
        intermediate_layer = nn.Linear(in_ch, out_ch)
        layers += [intermediate_layer, activation()]

    layers += [nn.Linear(*final_layer_definition), output_activation()]
    return nn.Sequential(*layers)


class GraphFeatureEncoder(torch.nn.Module):
    def __init__(
        self,
        in_features,
        conv_channels,
        num_heads,
        apply_batch_norm: int = True,
        ensure_trans_invar: bool = True,
        bias: bool = True,
        with_self_loops: bool = True,
    ):
        super().__init__()

        conv_params = dict(
            num_heads=num_heads,
            ensure_trans_invar=ensure_trans_invar,
            bias=bias,
            with_self_loops=with_self_loops,
        )
        self.apply_batch_norm = apply_batch_norm

        *first_conv_channels, final_conv_channel = conv_channels
        conv_layers = get_conv_layers(
            channels=[in_features] + conv_channels,
            conv=FeatureSteeredConvolution,
            conv_params=conv_params,
        )
        self.conv_layers = nn.ModuleList(conv_layers)

        self.batch_layers = [None for _ in first_conv_channels]
        if apply_batch_norm:
            self.batch_layers = nn.ModuleList(
                [nn.BatchNorm1d(channel) for channel in first_conv_channels]
            )

    def forward(self, x, edge_index):
        *first_conv_layers, final_conv_layer = self.conv_layers
        for conv_layer, batch_layer in zip(first_conv_layers, self.batch_layers):
            x = conv_layer(x, edge_index)
            x = F.relu(x)
            if batch_layer is not None:
                x = batch_layer(x)
        return final_conv_layer(x, edge_index)


class MeshSeg(torch.nn.Module):
    def __init__(
        self,
        in_features,
        encoder_features,
        num_classes,
        conv_channels,
        num_heads,
        apply_batch_norm=True,
    ):
        super().__init__()
        self.input_encoder = nn.Linear(in_features, encoder_features)
        self.gnn = GraphFeatureEncoder(
            in_features=encoder_features,
            conv_channels=conv_channels,
            num_heads=num_heads,
            apply_batch_norm=apply_batch_norm,
        )
        *_, final_conv_channel = conv_channels
        self.final_projection = nn.Linear(final_conv_channel, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.input_encoder(x)
        x = self.gnn(x, edge_index)
        return self.final_projection(x)


class MeshCorrSeg(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            num_seg_classes: int,
            encoder_channels: list,
            conv_channels: list,
            class_decoder_channels: list,
            assignment_decoder_channels: list,
            num_heads: int,
            apply_batch_norm: bool = True,
            sinkhorn_iterations: int = 5,
    ):
        super().__init__()
        self.sinkhorn_iterations = sinkhorn_iterations

        *_, final_conv_channel = conv_channels
        *_, final_encoder_channel = encoder_channels

        self.input_encoder = get_mlp_layers(
            channels=[in_features] + encoder_channels,
            activation=nn.ReLU)
        self.gnn = GraphFeatureEncoder(
            in_features=final_encoder_channel,
            conv_channels=conv_channels,
            num_heads=num_heads,
            apply_batch_norm=apply_batch_norm,
        )
        self.class_projection = get_mlp_layers(
            channels=[final_conv_channel] + class_decoder_channels + [num_seg_classes],
            activation=nn.ReLU,
        )
        self.assignment_projection = get_mlp_layers(
            channels=[final_conv_channel] + assignment_decoder_channels,
            activation=nn.ReLU,
        )

    def _compute_embedding(self, x, edge_index):
        return self.gnn(self.input_encoder(x), edge_index)

    def _compute_assignment(self, emb_s, emb_t):
        assignment_matrix = torch.einsum(
            'nd, md -> nm', self.assignment_projection(emb_s), self.assignment_projection(emb_t)
        )
        num_assignments = emb_s.shape[-1]
        assignment_matrix /= num_assignments ** 0.5
        assignment_matrix = solve_log_optimal_transport(
            assignment_matrix,
            num_iterations=self.sinkhorn_iterations,
        )
        return assignment_matrix

    def _compute_class_labels(self, emb_s, emb_t):
        return self.class_projection(emb_s), self.class_projection(emb_t)

    def forward(self, data_s, data_t):
        embeddings = [
            self._compute_embedding(x, edge_index)
            for x, edge_index in ((data_s.x, data_s.edge_index), (data_t.x, data_t.edge_index))
        ]
        assignment_matrix = self._compute_assignment(*embeddings)
        class_labels_s, class_labels_t = self._compute_class_labels(*embeddings)
        return assignment_matrix, class_labels_s, class_labels_t
