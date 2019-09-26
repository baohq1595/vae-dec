import torch
import torch.nn as nn
import math
from vae import VAE

class Clustering(nn.Module):
    def __init__(self, n_centroids, latent_dim):
        super(Clustering, self).__init__()
        self.n_centroids = n_centroids
        self.latent_dim = latent_dim
        self.theta_param = nn.Parameter(torch.ones(self.n_centroids, dtype=torch.float32) / self.n_centroids)
        self.mu_param = nn.Parameter(torch.zeros((self.latent_dim, self.n_centroids), dtype=torch.float32))
        self.lambda_param = nn.Parameter(torch.ones((self.latent_dim, self.n_centroids), dtype=torch.float32))

    def forward(self, z: torch.Tensor):
        batch_size = z.size()[0]
        temp_z = torch.transpose(z.repeat(self.n_centroids), 2, 1)
        temp_mu = self.mu_param
        temp_lambda = self.lambda_param
        temp_theta = self.theta_param * torch.ones(temp_mu.size())

        temp_p_c_z = torch.exp(
            torch.sum(
                torch.log(temp_theta) - 0.5 * torch.log(2 * math.pi * temp_lambda) - 
                (temp_z - temp_mu) ** 2 / (2 * temp_lambda),
                dim=0
            ) + 1e-10
        )

        return temp_p_c_z / torch.sum(temp_p_c_z, dim=-1, keepdim=True)

class ClusteringBasedVAE(nn.Module):
    def __init__(self, n_clusters, dimensions, **kwargs):
        super(ClusteringBasedVAE, self).__init__()
        self.vae = VAE(dimensions, kwargs)
        self.cluster = Clustering(n_clusters, dimensions[-1])

        
    def forward(self, x):
        x_decoded, z_mean = self.vae(x)
        cluster_output = self.cluster(x)

        return x_decoded, z_mean, cluster_output

    def train(self, dataloader):
        pass