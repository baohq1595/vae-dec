import torch
import torch.nn
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision import transforms
import itertools
from sklearn.mixture import GaussianMixture

from model.vae import *
from model.dec import *
from utils.utils import *
import matplotlib.pyplot as plt
from dataloader.metagenomics_dataset import GenomeDataset, GenomeDataset_v2
from transform.gene_transforms import numerize_genome_str
import torch.nn.functional as F

def train(model, train_dataloader, val_dataloader, optimizer, lr_scheduler=None, likeliood=F.mse_loss, is_continue=False, **params):
    num_pretrained_epoch = params.get('pretrained_epochs', 10)
    save_path = params.get('save_path', 'output/model/vae_model.pt')
    dataset_name = params.get('dataset_name', '')
    log_visualize = params.get('log_visualize', False)

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        
        if not is_continue:
            return
    else:
        os.makedirs(os.path.dirname(save_path))

    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    model = model.to(device)

    print('Pretrains VAE using only reconstruction loss...')
    for pre_epoch in range(num_pretrained_epoch):
        total_loss = 0.0
        iters = 0
        for i, data in enumerate(train_dataloader):
            x = data[0]

            # Flatten 28x28 to 1x784 on mnist dataset
            if dataset_name == 'mnist':
                x = x.view(x.size()[0], -1)
            
            x = x.float()
            x = x.to(device)
            # Forward pass
            z, mu, log_var = model.encoder(x)
            x_decoded = model.decoder(z)

            # Loss is likelihood loss and kld loss ~ standard normal distribution as prior
            likeliood_loss = likeliood(x_decoded, x)
            kld_loss = model._kld(z, (mu, log_var), None)
            loss = likeliood_loss + kld_loss 

            # Calculate gradient and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().numpy()

            iters += 1
        
        print('VAE resconstruction loss: ', total_loss / iters)
        steplr.step()

    Z = []
    Y = []
    with torch.no_grad():
        for x, y in train_dataloader:
            # Flatten 28x28 to 1x784 on mnist dataset
            if dataset_name == 'mnist':
                x = x.view(x.size()[0], -1)

            x = x.float()
            x = x.to(device)
            z, mu, log_var = model.encoder(x)
            assert F.mse_loss(mu, log_var) == 0
            Z.append(mu)
            Y.append(y)

    Z = torch.cat(Z, 0).detach().cpu().numpy()
    Y = torch.cat(Y, 0).detach().numpy()
    gmm = GaussianMixture(n_components=model.n_centroids, covariance_type='diag')
    predict = gmm.fit_predict(Z)
    

    print('Accuracy = {:.4f}%'.format(cluster_accuracy(predict, Y)[0] * 100))

    model.mu_c.data = torch.from_numpy(gmm.means_).to(device).float()
    model.log_sigma_c.data = torch.log(torch.from_numpy(gmm.covariances_).to(device).float())
    model.pi.data = torch.from_numpy(gmm.weights_).to(device).float()

    torch.save(model.state_dict(), pretrained_save_path)