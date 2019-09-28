import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import os
from time import gmtime, strftime
import numpy as np

from model.vae import VAE
from utils.utils import cluster_accuracy

class Clustering(nn.Module):
    def __init__(self, n_centroids, latent_dim):
        super(Clustering, self).__init__()
        self.n_centroids = n_centroids
        self.latent_dim = latent_dim
        self.theta_param = nn.Parameter(torch.ones(self.n_centroids, dtype=torch.float32) / self.n_centroids)
        self.mu_param = nn.Parameter(torch.zeros((self.latent_dim, self.n_centroids), dtype=torch.float32))
        self.lambda_param = nn.Parameter(torch.ones((self.latent_dim, self.n_centroids), dtype=torch.float32))
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    def forward(self, z: torch.Tensor):
        '''
        TODO Add formula for calculating gamma ~ q(c|x)
        '''
        batch_size = z.size()[0]
        temp_z = torch.transpose(z.repeat(self.n_centroids, 1, 1), 0, 1)

        # Add 1 dimension to self.mu_param, self.lambda_param
        temp_mu = (self.mu_param[None, :, :]).repeat(batch_size, 1, 1)
        temp_lambda = (self.lambda_param[None, :, :]).repeat(batch_size, 1, 1)

        # Add 2 dimensions to self.theta_param
        temp_theta = self.theta_param[None, None, :] * torch.ones(temp_mu.size()).to(self.device)

        temp_p_c_z = torch.exp(
            torch.sum(
                torch.log(temp_theta) - 0.5 * math.log(2 * math.pi) * temp_lambda - 
                (temp_z - temp_mu) ** 2 / (2 * temp_lambda),
                dim=1
            )
        ) + 1e-10

        return temp_p_c_z / torch.sum(temp_p_c_z, dim=-1, keepdim=True)

class ClusteringBasedVAE(nn.Module):
    def __init__(self, n_clusters, dimensions, alpha, **kwargs):
        super(ClusteringBasedVAE, self).__init__()
        self.vae = VAE(dimensions, **kwargs)
        self.latent_dim = dimensions[-1]
        self.cluster = Clustering(n_clusters, self.latent_dim)
        self.is_logits = kwargs.get('logits', True)
        self.alpha = alpha
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        
        if self.is_logits:
            self.resconstruction_loss = nn.modules.loss.MSELoss()
        else:
            self.resconstruction_loss = nn.modules.loss.BCELoss()

        self.models = nn.ModuleList()
        self.models.append(self.vae)
        self.models.append(self.cluster)

        
    def forward(self, x):
        x_decoded, latent, z_mean, z_log_var = self.vae(x)
        cluster_output = self.cluster(latent)

        return x_decoded, latent, z_mean, z_log_var, cluster_output

    def criterion(self, x: torch.Tensor, x_decoded: torch.Tensor, z: torch.Tensor,
                    z_mean: torch.Tensor, z_log_var: torch.Tensor):
        '''
        TODO Should put the formula here
        '''
        batch_size = x.size()[0]
        temp_z = torch.transpose(z.repeat(self.cluster.n_centroids, 1, 1), 0, 1)
        temp_z_mean = torch.transpose(z_mean.repeat(self.cluster.n_centroids, 1, 1), 0, 1)
        temp_z_log_var = torch.transpose(z_log_var.repeat(self.cluster.n_centroids, 1, 1), 0, 1)

        # Add 1 dimension to self.mu_param, self.lambda_param, 2 to self.theta_param
        temp_mu = self.cluster.mu_param[None, :, :].repeat(batch_size, 1, 1)
        temp_lambda = self.cluster.lambda_param[None, :, :].repeat(batch_size, 1, 1)
        temp_theta = self.cluster.theta_param[None, None, :] * torch.ones(
                            batch_size, self.latent_dim, self.cluster.n_centroids).to(self.device)

        p_c_z = torch.exp(
            torch.sum(
                torch.log(temp_theta) - 0.5 * torch.log(2 * math.pi * temp_lambda) - 
                (temp_z - temp_mu) ** 2 / (2 * temp_lambda),
                dim=1
            )
        ) + 1e-10

        q_c_x = p_c_z / torch.sum(p_c_z, dim=-1, keepdim=True)
        gamma_t = q_c_x.repeat(self.latent_dim, 1, 1).transpose(0, 1)

        loss = self.alpha * self.resconstruction_loss(x_decoded, x) + \
            torch.sum(
                0.5 * gamma_t * (self.latent_dim * math.log(math.pi * 2)) +
                torch.log(temp_lambda) + torch.exp(temp_z_log_var) / temp_lambda +
                (temp_z_mean - temp_mu) ** 2 / temp_lambda,
                dim=[1, 2]
            )\
            - 0.5 * torch.sum(z_log_var + 1, dim=-1)\
            - torch.sum(torch.log(self.cluster.theta_param[None,:].repeat(batch_size, 1, 1)) * q_c_x, dim=-1)\
            + torch.sum(torch.log(q_c_x) * q_c_x, dim=-1)
        
        return loss.mean()

    def __setup_device(self, device):
        # for module in self.models:
        #     module.to(device)
        self.vae = self.vae.to(device)
        self.cluster = self.cluster.to(device)

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
                **params):
        optimizer = torch.optim.Adam(self.models.parameters(), lr=0.002, eps=1e-4)
        num_epochs = params.get('epochs', 10)
        save_path = params.get('save_path', 'output/model')
        dataset_name = params.get('dataset_name', '')

        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        self.__setup_device(self.device)

        for epoch in range(num_epochs):
            for i, data in enumerate(train_dataloader):
                self.models.zero_grad()

                # Get only data, ignore label (data[1])
                x = data[0]

                # Flatten 28x28 to 1x784 on mnist dataset
                if dataset_name == 'mnist':
                    x = x.view(x.size()[0], -1)
                
                x = x.to(self.device)

                # Forward thru vae model
                x_decoded, latent, z_mean, z_log_var = self.vae(x)

                # Forward to clustering model
                gamma = self.cluster(latent)

                # Acquire the loss
                loss = self.criterion(x, x_decoded, latent, z_mean, z_log_var)

                # Calculate gradients
                loss.backward()

                # Update models
                optimizer.step()

            # For each epoch, log the p_c_z accuracy
            with torch.no_grad():
                mean_accuracy = 0.0
                iters = 0
                for i, data in enumerate(val_dataloader):
                    # Get z value
                    x = data[0].to(self.device)
                    labels = data[1].cpu().detach().numpy()
                    if dataset_name == 'mnist':
                        x = x.view(x.size()[0], -1)
                    
                    x_decoded, latent, z_mean, z_log_var = self.vae(x)

                    # Cluster latent space
                    gamma = self.cluster(latent)
                    sample = np.argmax(gamma.cpu().detach().numpy(), axis=1)
                    mean_accuracy += cluster_accuracy(sample, labels)[0]
                    iters += 1

                print('accuracy p(c|z): %0.8f' % (mean_accuracy / iters))

        torch.save(self.models.state_dict(), os.path.join(save_path, 'vae-dec-model-{}'
            .format(strftime("%Y-%m-%d-%H-%M", gmtime())
        )))

