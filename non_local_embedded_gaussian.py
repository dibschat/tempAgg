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

        # theta = model.ConvNd(in_blob1, prefix + '_theta', in_dim1, latent_dim, [1, 1, 1],
        #                      strides=[1, 1, 1], pads=[0, 0, 0] * 2, **init_params1)
        self.theta = nn.Conv1d(in_channels=self.in_dim1, out_channels=self.latent_dim,
                               kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.theta.weight, mean=0, std=0.01)
        nn.init.constant_(self.theta.bias, 0)

        # phi = model.ConvNd( in_blob2, prefix + '_phi', in_dim2, latent_dim, [1, 1, 1],
        #                     strides=[1, 1, 1], pads=[0, 0, 0] * 2, **init_params1)
        self.phi = nn.Conv1d(in_channels=self.in_dim2, out_channels=self.latent_dim,
                             kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.phi.weight, mean=0, std=0.01)
        nn.init.constant_(self.phi.bias, 0)

        # g = model.ConvNd( in_blob2, prefix + '_g', in_dim2, latent_dim, [1, 1, 1],
        #                   strides=[1, 1, 1], pads=[0, 0, 0] * 2, **init_params1)
        self.g = nn.Conv1d(in_channels=self.in_dim2, out_channels=self.latent_dim,
                           kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.g.weight, mean=0, std=0.01)
        nn.init.constant_(self.g.bias, 0)

        # if cfg.FBO_NL.SCALE: theta_phi = model.Scale( theta_phi, theta_phi, scale=latent_dim**-.5)
        if self.scale:
            self.scale_factor = torch.tensor([self.latent_dim ** self.scale_factor], requires_grad=True).to('cuda')

        # """Pre-activation style non-linearity."""
        # x = model.LayerNorm( x, [x + "_ln", x + "_ln_mean", x + "_ln_std"])[0]
        # model.Relu(x, x + "_relu")
        # blob_out = model.ConvNd( blob_out, prefix + '_out', latent_dim, in_dim1, [1, 1, 1],
        #                          strides=[1, 1, 1], pads=[0, 0, 0] * 2,  **init_params2)
        # blob_out = model.Dropout( blob_out, blob_out + '_drop', ratio= 0.2, is_test=False)
        self.final_layers = nn.Sequential(
            nn.LayerNorm(torch.Size([self.latent_dim, self.video_feat_dim])),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.latent_dim, out_channels=self.in_dim1, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=self.dropout_rate),
        )

    def forward(self, x_past, x_curr):
        # theta = model.ConvNd(in_blob1, prefix + '_theta', in_dim1, latent_dim, [1, 1, 1],
        #                      strides=[1, 1, 1], pads=[0, 0, 0] * 2, **init_params1)
        # theta, theta_shape_5d = model.Reshape(theta, [theta + '_re', theta + '_shape5d'],
        #                                       shape=(-1, latent_dim, num_feat1))
        theta_x = self.theta(x_curr)
        theta_x = theta_x.permute(0, 2, 1)

        # phi = model.ConvNd( in_blob2, prefix + '_phi', in_dim2, latent_dim, [1, 1, 1],
        #                     strides=[1, 1, 1], pads=[0, 0, 0] * 2, **init_params1)
        # phi, phi_shape_5d = model.Reshape(phi, [phi + '_re', phi + '_shape5d'], shape=(-1, latent_dim, num_feat2))
        phi_x = self.phi(x_past)

        # g = model.ConvNd( in_blob2, prefix + '_g', in_dim2, latent_dim, [1, 1, 1],
        #                   strides=[1, 1, 1], pads=[0, 0, 0] * 2, **init_params1)
        # g, g_shape_5d = model.Reshape(g, [g + '_re', g + '_shape5d'], shape=(-1, latent_dim, num_feat2))
        g_x = self.g(x_past)
        g_x = g_x.permute(0, 2, 1)

        # (N, C, num_feat1), (N, C, num_feat2) -> (N, num_feat1, num_feat2)
        # theta_phi = model.net.BatchMatMul( [theta, phi], prefix + '_affinity', trans_a=1)
        theta_phi = torch.matmul(theta_x, phi_x)

        # if cfg.FBO_NL.SCALE: theta_phi = model.Scale( theta_phi, theta_phi, scale=latent_dim**-.5)
        if self.scale:
            theta_phi = theta_phi * self.scale_factor

        # p = model.Softmax(theta_phi, theta_phi + '_prob', engine='CUDNN', axis=2)
        p_x = F.softmax(theta_phi, dim=-1)

        # (N, C, num_feat2), (N, num_feat1, num_feat2) -> (B, C, num_feat1)
        # t = model.net.BatchMatMul([g, p], prefix + '_y', trans_b=1)
        t_x = torch.matmul(p_x, g_x)

        # blob_out, t_shape = model.Reshape( [t, theta_shape_5d], [t + '_re', t + '_shape3d'])
        t_x = t_x.permute(0, 2, 1).contiguous()

        # """Pre-activation style non-linearity."""
        # x = model.LayerNorm( x, [x + "_ln", x + "_ln_mean", x + "_ln_std"])[0]
        # model.Relu(x, x + "_relu")
        # blob_out = model.ConvNd( blob_out, prefix + '_out', latent_dim, in_dim1, [1, 1, 1], strides=[1, 1, 1],
        #                          pads=[0, 0, 0] * 2,  **init_params2)
        # blob_out = model.Dropout( blob_out, blob_out + '_drop', ratio= 0.2, is_test=False)
        W_t = self.final_layers(t_x)

        z_x = W_t + x_curr
        return z_x
