import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import os
from time import gmtime, strftime
import numpy as np

import torch.nn.functional as F

from model.vae import VAE
from utils.utils import cluster_accuracy
from utils.distributions import log_gaussian


class ClusteringBasedVAE(nn.Module):
    def __init__(self, n_clusters, dimensions, alpha, **kwargs):
        super(ClusteringBasedVAE, self).__init__()
        self.vae = VAE(dimensions, **kwargs)
        self.embedding_dim = dimensions[0]
        self.latent_dim = dimensions[-1]
        self.is_logits = kwargs.get('logits', False)
        self.alpha = alpha

        self.n_centroids = n_clusters
        self.theta_param = nn.Parameter(torch.ones(self.n_centroids, dtype=torch.float32) / self.n_centroids)
        self.mu_param = nn.Parameter(torch.zeros((self.n_centroids, self.latent_dim), dtype=torch.float32))
        self.lambda_param = nn.Parameter(torch.ones((self.n_centroids, self.latent_dim), dtype=torch.float32))
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        
        if self.is_logits:
            self.resconstruction_loss = nn.modules.loss.MSELoss()
        else:
            self.resconstruction_loss = self.binary_cross_entropy

        self.models = nn.ModuleList()

    def binary_cross_entropy(self, x, r):
        return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)
        
    def forward(self, x):
        x_decoded, latent, z_mean, z_log_var = self.vae(x)

        pi = self.theta_param
        log_sigmac_c = self.lambda_param
        mu_c = self.mu_param
        pzc = torch.exp(torch.log(pi.unsqueeze(0)) + self.log_gaussians(latent, mu_c, log_sigmac_c))

        return pzc

    def elbo_loss(self, x, L=1):
        det = 1e-10
        res_loss = 0.0

        z, mu, logvar = self.vae.encoder(x)
        for l in range(L):
            z = torch.randn_like(mu) * torch.exp(logvar/2) + mu

            x_decoded = self.vae.decoder(z)
            res_loss += F.binary_cross_entropy(x_decoded, x)

        res_loss /= L
        loss = res_loss * x.size(1)
        pi = self.theta_param
        log_sigma2_c = self.lambda_param
        mu_c = self.mu_param

        z = torch.randn_like(mu) * torch.exp(logvar / 2) + mu
        pcz = torch.exp(torch.log(pi.unsqueeze(0)) + self.log_gaussians(z, mu_c, log_sigma2_c)) + det

        pcz = pcz / (pcz.sum(1).view(-1, 1)) # batch_size*clusters

        loss += 0.5 * torch.mean(torch.sum(pcz * torch.sum(log_sigma2_c.unsqueeze(0) + 
                    torch.exp(logvar.unsqueeze(1) - log_sigma2_c.unsqueeze(0)) +
                    (mu.unsqueeze(1) - mu_c.unsqueeze(0))**2 / torch.exp(log_sigma2_c.unsqueeze(0)), 2), 1))

        loss -= torch.mean(torch.sum(pcz * torch.log(pi.unsqueeze(0) / (pcz)), 1)) + 0.5 * torch.mean(torch.sum(1 + logvar, 1))

        return loss        

    def log_gaussians(self, x, mus, logvars):
        G = []
        for c in range(self.n_centroids):
            G.append(log_gaussian(x, mus[c:c + 1, :], logvars[c:c + 1,:]).view(-1, 1))
        
        return torch.cat(G, 1)


    def criterion(self, x: torch.Tensor, x_decoded: torch.Tensor, z: torch.Tensor,
                    z_mean: torch.Tensor, z_log_var: torch.Tensor, gamma: torch.Tensor):
        '''
        TODO Should put the formula here
        '''
        batch_size = x.size()[0]

        temp_z = z.repeat(self.cluster.n_centroids, 1, 1).permute(1, 2, 0)
        temp_z_mean = z_mean.repeat(self.cluster.n_centroids, 1, 1).permute(1, 2, 0)
        temp_z_log_var = z_log_var.repeat(self.cluster.n_centroids, 1, 1).permute(1, 2, 0)

        # Add 1 dimension to self.mu_param, self.lambda_param, 2 to self.theta_param
        temp_mu = self.cluster.mu_param[None, :, :].repeat(batch_size, 1, 1)
        temp_lambda = self.cluster.lambda_param[None, :, :].repeat(batch_size, 1, 1)
        temp_theta = self.cluster.theta_param[None, None, :] * torch.ones(
                            batch_size, self.latent_dim, self.cluster.n_centroids).to(self.device)

        gamma_t = gamma.repeat(self.latent_dim, 1, 1).permute(1, 0, 2)

        loss = self.alpha * self.embedding_dim * self.resconstruction_loss(x_decoded, x) + \
            torch.sum(
                0.5 * gamma_t * (self.latent_dim * math.log(math.pi * 2) +
                torch.log(temp_lambda) + torch.exp(temp_z_log_var) / temp_lambda +
                (temp_z_mean - temp_mu) ** 2 / temp_lambda),
                dim=[1, 2]
            )\
            - 0.5 * torch.sum(z_log_var + 1, dim=-1)\
            - torch.sum(torch.log(self.cluster.theta_param[None,:].repeat(batch_size, 1, 1)) * gamma, dim=-1)\
            + torch.sum(torch.log(gamma) * gamma, dim=-1)

        return loss.mean()

    def __setup_device(self, device):
        self.models = self.models.to(device)
