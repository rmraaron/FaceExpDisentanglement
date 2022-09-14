import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from Model.losses import kl_divergence_loss, kl_info_bottleneck_reg_mws_id_exp


class VariationalAE(nn.Module):
    def __init__(self, args, vertices_num, latent_channels_id=160, latent_channels_exp=160):
        super().__init__()
        self.args = args
        self.vertices_num = vertices_num
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.maxp = nn.MaxPool1d(vertices_num)

        self.fc_id_mu = nn.Linear(1024, latent_channels_id)
        self.fc_id_sigma = nn.Linear(1024, latent_channels_id)
        self.fc_exp_mu = nn.Linear(1024, latent_channels_exp)
        self.fc_exp_sigma = nn.Linear(1024, latent_channels_exp)

        self.fc1_id = nn.Linear(latent_channels_id, 256)
        self.fc2_id = nn.Linear(256, vertices_num * 3)

        self.fc1_exp = nn.Linear(latent_channels_exp, 256)
        self.fc2_exp = nn.Linear(256, vertices_num * 3)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def feature_extraction_layers(self, x):
        x = x.transpose(2, 1).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.maxp(x)
        x = x.view(-1, 1024)

        return x

    def decoding(self, z_id, z_exp):
        batch_size = z_id.size(0)

        x_hat_id = self.fc2_id(F.relu(self.fc1_id(z_id)))
        x_hat_exp = self.fc2_exp(F.relu(self.fc1_exp(z_exp)))

        x_hat_id = x_hat_id.view(batch_size, 3, self.vertices_num).transpose(1, 2).contiguous()
        x_hat_exp = x_hat_exp.view(batch_size, 3, self.vertices_num).transpose(1, 2).contiguous()
        x_hat = torch.add(x_hat_id, x_hat_exp)

        return x_hat, x_hat_id, x_hat_exp

    def forward(self, x1, neutral_mask=None):
        x = self.feature_extraction_layers(x1)

        id_mu = self.fc_id_mu(x)
        id_sigma = self.fc_id_sigma(x)
        exp_mu = self.fc_exp_mu(x)
        exp_sigma = self.fc_exp_sigma(x)

        if self.args.dataset == "COMA" and self.args.info_bn:
            z_id, z_exp, kl_loss_id, kl_loss_exp = \
                kl_info_bottleneck_reg_mws_id_exp(
                    id_mu, id_sigma, exp_mu, exp_sigma, neutral_mask=neutral_mask)
            kl_loss_id /= z_id.shape[1]
            kl_loss_exp /= z_exp.shape[1]
        else:
            z_id, kl_loss_id = kl_divergence_loss(id_mu, id_sigma)
            z_exp, kl_loss_exp = kl_divergence_loss(exp_mu, exp_sigma)

        kl_loss_id = kl_loss_id.mean()
        kl_loss_exp = kl_loss_exp.mean()

        x_hat, x_hat_id, x_hat_exp = self.decoding(z_id, z_exp)

        return x_hat, x_hat_id, x_hat_exp, kl_loss_id, kl_loss_exp, z_id, z_exp

    def encoding_eval(self, x):
        x = self.feature_extraction_layers(x)

        id_mu = self.fc_id_mu(x)
        exp_mu = self.fc_exp_mu(x)

        return id_mu, exp_mu
