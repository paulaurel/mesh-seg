from itertools import tee

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from .graph_conv.feat_steer_conv import FeatureSteeredConvolution


def pairwise(iterable):
    """Iterate over all pairs of consecutive items in a list.

    Notes
    -----
        [s0, s1, s2, s3, ...] -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_conv_layers(channels: list, conv: MessagePassing, conv_params: dict):
    """Define basic multilayered graph convolution network architecture.

    Parameters
    ----------
    channels: list
        List of integers specifying the convolution channels.
    conv: torch.nn.Module
        Convolution layer.
    conv_params: dict
        Dictionary specifying convolution parameters.

    Returns
    -------
    list
        List of convolutions with the specified channels.
    """
    conv_layers = [
        conv(in_ch, out_ch, **conv_params) for in_ch, out_ch in pairwise(channels)
    ]
    return conv_layers


def get_mlp_layers(channels: list, activation, output_activation=nn.Identity):
    """Define basic multilayered perceptron network."""
    layers = []
    *intermediate_layer_definitions, final_layer_definition = pairwise(channels)

    for in_ch, out_ch in intermediate_layer_definitions:
        intermediate_layer = nn.Linear(in_ch, out_ch)
        layers += [intermediate_layer, activation()]

    layers += [nn.Linear(*final_layer_definition), output_activation()]
    return nn.Sequential(*layers)


class GraphFeatureEncoder(torch.nn.Module):
    """Graph neural network consisting of sequentially stacked graph convolutions."""
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
        conv_channels,
        encoder_channels,
        decoder_channels,
        num_classes,
        num_heads,
        apply_batch_norm=True,
    ):
        super().__init__()
        self.input_encoder = get_mlp_layers(
            channels=[in_features] + encoder_channels,
            activation=nn.ReLU,
        )
        self.gnn = GraphFeatureEncoder(
            in_features=encoder_features,
            conv_channels=conv_channels,
            num_heads=num_heads,
            apply_batch_norm=apply_batch_norm,
        )
        *_, final_conv_channel = conv_channels

        self.final_projection = get_mlp_layers(
            [final_conv_channel] + decoder_channels + [num_classes],
            activation=nn.ReLU,
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.input_encoder(x)
        x = self.gnn(x, edge_index)
        return self.final_projection(x)
