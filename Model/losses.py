import math

import torch
from torch import nn
import numpy as np


def kl_divergence_loss(mu, sigma):

    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both).
    std = torch.exp(sigma / 2)
    p_z = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q_zx = torch.distributions.Normal(mu, std)

    # Sample the latent code z.
    z = q_zx.rsample()

    # 2. get the probabilities from the equation
    log_q_zx = q_zx.log_prob(z)
    log_p_z = p_z.log_prob(z)

    # kl
    kl = log_q_zx - log_p_z
    kl = kl.sum(-1)

    return z, kl


def log_sum_exp(value, dim, keepdim=False):
    """ Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.sum(torch.exp(value0),
                                   dim=dim, keepdim=keepdim))


def kl_info_bottleneck_reg_mws_id_exp(mu_id, log_var_id, mu_exp, log_var_exp,
                                      dataset_size=18424, beta=50., neutral_mask=None):
    """
    Returns KL loss plus information bottleneck regulariser.

    If neutral_mask is given, neutral faces' id latent code will not be penalised with this regulariser term.

    :param mu_id: Identity latent code's mean.
    :param log_var_id: Identity latent code's log variance.
    :param mu_exp: Expression latent code's mean.
    :param log_var_exp: Expression latent code's log variance.
    :param dataset_size: Total data points in the dataset, i.e. how many faces in your training set.
    :param beta: Weighting factor to strengthen/weaken the effect of the information bottleneck regularier.
                 The larger, the stronger.
    :param neutral_mask: Mask for the batch. True when input is neutral, False when input is with expression.

    :return: Sampled id latent code, Sampled expression latent code,
            kl loss + information regulariser (not averaged, all latent code are summed)
    """

    batch_size = mu_id.shape[0]

    sigma_id = torch.exp(log_var_id / 2)
    sigma_exp = torch.exp(log_var_exp / 2)
    q_z_id_x_dist = torch.distributions.Normal(mu_id, sigma_id)  # Predicted ID distribution.
    q_z_exp_x_dist = torch.distributions.Normal(mu_exp, sigma_exp)  # Predicted ID distribution.
    p_id_dist = torch.distributions.Normal(torch.zeros_like(mu_id),
                                           torch.ones_like(sigma_id))  # Prior distribution.
    p_exp_dist = torch.distributions.Normal(torch.zeros_like(mu_exp),
                                            torch.ones_like(sigma_exp))  # Prior distribution.

    z_id = q_z_id_x_dist.rsample()
    z_exp = q_z_exp_x_dist.rsample()

    kl_loss_id = torch.distributions.kl_divergence(q_z_id_x_dist, p_id_dist).sum(1)
    kl_loss_exp = torch.distributions.kl_divergence(q_z_exp_x_dist, p_exp_dist).sum(1)

    if beta == 0.:
        return z_id, z_exp, kl_loss_id, kl_loss_exp

    # Likelihood: log q(z|x).
    log_q_z_id_x = q_z_id_x_dist.log_prob(z_id).sum(1)

    # Mini-batch weighted sampling.
    _log_q_z_id = q_z_id_x_dist.log_prob(z_id.unsqueeze(1))
    log_q_z_id = (log_sum_exp(_log_q_z_id.sum(2), dim=1, keepdim=False) - math.log(
        batch_size * dataset_size))

    kl_info_bottleneck_reg_id = kl_loss_id
    if neutral_mask is not None:
        exp_mask = torch.logical_not(neutral_mask)
        kl_info_bottleneck_reg_id[exp_mask] += beta * (log_q_z_id_x - log_q_z_id)[exp_mask]
    else:
        kl_info_bottleneck_reg_id += beta * (log_q_z_id_x - log_q_z_id)

    return z_id, z_exp, kl_info_bottleneck_reg_id, kl_loss_exp
