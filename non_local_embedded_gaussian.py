# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F


class NONLocalBlock1D(nn.Module):
    def __init__(self, args, recent_dim, spanning_dim, latent_dim):
        super(NONLocalBlock1D, self).__init__()

        self.in_dim1 = recent_dim
        self.in_dim2 = spanning_dim

        self.scale = args.scale
        self.scale_factor = args.scale_factor

        self.dropout_rate = args.dropout_rate

        self.latent_dim = latent_dim
        self.video_feat_dim = args.video_feat_dim

        self.theta = nn.Conv1d(in_channels=self.in_dim1, out_channels=self.latent_dim,
                               kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.theta.weight, mean=0, std=0.01)
        nn.init.constant_(self.theta.bias, 0)

        self.phi = nn.Conv1d(in_channels=self.in_dim2, out_channels=self.latent_dim,
                             kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.phi.weight, mean=0, std=0.01)
        nn.init.constant_(self.phi.bias, 0)

        self.g = nn.Conv1d(in_channels=self.in_dim2, out_channels=self.latent_dim,
                           kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.g.weight, mean=0, std=0.01)
        nn.init.constant_(self.g.bias, 0)

        if self.scale:
            self.scale_factor = torch.tensor([self.latent_dim ** self.scale_factor], requires_grad=True).to('cuda')

        # """Pre-activation style non-linearity."""
        self.final_layers = nn.Sequential(
            nn.LayerNorm(torch.Size([self.latent_dim, self.video_feat_dim])),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.latent_dim, out_channels=self.in_dim1, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=self.dropout_rate),
        )

    def forward(self, x_past, x_curr):
        theta_x = self.theta(x_curr)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x_past)

        g_x = self.g(x_past)
        g_x = g_x.permute(0, 2, 1)

        # (N, C, num_feat1), (N, C, num_feat2) -> (N, num_feat1, num_feat2)
        theta_phi = torch.matmul(theta_x, phi_x)

        if self.scale:
            theta_phi = theta_phi * self.scale_factor

        p_x = F.softmax(theta_phi, dim=-1)

        # (N, C, num_feat2), (N, num_feat1, num_feat2) -> (B, C, num_feat1)
        t_x = torch.matmul(p_x, g_x)

        t_x = t_x.permute(0, 2, 1).contiguous()
        
        W_t = self.final_layers(t_x)

        z_x = W_t + x_curr
        return z_x
