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

import copy, math
from tqdm import tqdm

from tensorboardX import SummaryWriter
from time import gmtime, strftime

def train(n_centroids, model, train_dataloader, val_dataloader, optimizer,
            lr_scheduler=None, likelihood=F.mse_loss, is_continue=False, retrain=False, **params):
    num_pretrained_epoch = params.get('pretrained_epochs', 10)
    save_path = params.get('save_path', 'output/model/vae_model.pt')
    dataset_name = params.get('dataset_name', '')
    log_visualize = params.get('log_visualize', False)
    log_dir = params.get('log_dir', None)

    if log_visualize:
        if log_dir is None:
            log_dir = 'report/log/{}.'.format(strftime("%Y-%m-%d-%H-%M-%S", gmtime()))

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        summary_writer = SummaryWriter(log_dir)

    if not retrain:
        if os.path.exists(save_path):
            print('Loading pretrained vae model...')
            model.load_state_dict(torch.load(save_path))
            
            if not is_continue:
                print('Finished.')
                return
            print('Continue training process...')
    else:
        print('Starts training VAE from scratch.')

    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    model = model.to(device)

    best_acc = 0.0
    best_model = None

    print('Pretrains VAE using only reconstruction loss...')
    for pre_epoch in tqdm(range(num_pretrained_epoch)):
        total_loss = 0.0
        total_likelihood_loss = 0.0
        total_kld_loss = 0.0
        iters = 0
        for i, data in tqdm(enumerate(train_dataloader)):
            x = data[0]

            # Flatten 28x28 to 1x784 on mnist dataset
            if dataset_name == 'mnist':
                x = x.view(x.size()[0], -1)
            
            x = x.float()
            x = x.to(device)

            # Forward pass
            x_decoded, z, mu, log_var = model(x)

            # Loss is likelihood loss and kld loss ~ standard normal distribution as prior
            likelihood_loss = likelihood(x_decoded, x)
            kld_loss = torch.sum(model._kld(z, (mu, log_var), None))
            loss = likelihood_loss + kld_loss 

            # Calculate gradient and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().numpy()
            total_likelihood_loss += likelihood_loss.detach().cpu().numpy()
            total_kld_loss += kld_loss.detach().cpu().numpy()

            iters += 1
        
        total_likelihood_loss = total_likelihood_loss / iters
        total_kld_loss = total_kld_loss / iters
        total_loss = total_loss / iters

        # Add to tensorboard
        if log_visualize:
            summary_writer.add_scalar('likelihood_loss', total_likelihood_loss, pre_epoch)
            summary_writer.add_scalar('kld_loss', total_kld_loss, pre_epoch)
            summary_writer.add_scalar('total_loss', total_loss, pre_epoch)

        print('VAE likelihood loss: ', total_likelihood_loss)
        print('VAE kld loss: ', total_kld_loss)

        if lr_scheduler:
            lr_scheduler.step()

        if pre_epoch % 5 == 0:
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
                    Z.append(mu)
                    Y.append(y)

            Z = torch.cat(Z, 0).detach().cpu().numpy()
            Y = torch.cat(Y, 0).detach().numpy()
            gmm = GaussianMixture(n_components=n_centroids, covariance_type='diag')
            predict = gmm.fit_predict(Z)

            accuracy = cluster_accuracy(predict, Y)[0] * 500

            print('Accuracy = {:.4f}%'.format(accuracy))
            if accuracy > best_acc:
                best_acc = accuracy
                best_model = copy.deepcopy(model)

    
            if log_visualize:
                summary_writer.add_scalar('Accuracy', accuracy)

    sub_save_dir = 'vae-{}'.format(strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
    save_dir = os.path.join(save_path, sub_save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(best_model.state_dict(), os.path.join(save_dir, 'vae.pt'))

def get_metagenomics_dataloader(data_path):
    gen_transforms = transforms.Compose([
        numerize_genome_str,
        transforms.ToTensor()
    ])
    genomics_dataset = GenomeDataset_v2(data_path, return_raw=False)
    dataloader = DataLoader(genomics_dataset, batch_size=32, shuffle=True)

    return dataloader

if __name__ == "__main__":
    import sys
    sys.path.append('.')
    dimensions = [256, 500, 500, 2000, 10]
    model_params = {
        'decoder_final_activation': '',
        'pretrained_epochs': 100,
        'epochs': 1,
        'save_path': 'output/model',
        'dataset_name': 'gene',
        'logits': True
    }

    vae = VAE(dimensions, **model_params)

    gen_dataloader = get_metagenomics_dataloader('data/gene/L1.fna')

    if torch.cuda.is_available():
        print('Cuda is available')
        vae = vae.cuda()    
    else:
        print('No GPU')

    optimizer = torch.optim.SGD(vae.parameters(), lr=0.001, momentum=0.9)
    train(2, vae, gen_dataloader, gen_dataloader, optimizer, retrain=True, **model_params)
    # train(dec_cluster, gen_dataloader, gen_dataloader, **model_params)