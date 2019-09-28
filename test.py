import torch
import torch.nn
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision import transforms

from model.vae import *
from model.dec import *

if __name__ == '__main__':
    dimensions = [784, 500, 500, 2000, 10]
    model_params = {
        'decoder_final_activation': 'sigmoid',
        'epochs': 20,
        'save_path': 'output/model',
        'dataset_name': 'mnist'
    }

    dec_cluster = ClusteringBasedVAE(10, dimensions, 1, **model_params)

    train_dataloader = DataLoader(MNIST('data/mnist', train=True, 
                                        download=True,
                                        transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.1307,), (0.3081,))
                                                ])),
                                batch_size=32,
                                shuffle=True)

    val_dataloader = DataLoader(MNIST('data/mnist', train=False, 
                                        download=True,
                                        transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.1307,), (0.3081,))
                                                ])),
                                batch_size=32,
                                shuffle=True)

    dec_cluster.train(train_dataloader, val_dataloader, **model_params)