from itertools import tee

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def get_conv_layers(channels: list, conv, conv_params):
    """Define basic multilayered graph convolution network architecture."""
    conv_layers = [
        conv(in_ch, out_ch, **conv_params) for in_ch, out_ch in pairwise(channels)
    ]
    return conv_layers


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

    def _get_encoding(self, x, edge_index):
        return self.gnn(self.input_encoder(x), edge_index)

    def forward(self, data):
        ref_data, edge_index_ref = data.ref, data.edge_index_ref
        query_data, edge_index_query = data.query, data.edge_index_query

        ref_encoding = self._get_encoding(ref_data, edge_index_ref)
        query_encoding = self._get_encoding(query_data, edge_index_query)

        assignment_matrix = torch.einsum('bdn,bdm->bnm', ref_encoding, query_encoding)

        return self.final_projection(x)
