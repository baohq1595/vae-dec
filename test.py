import torch
import torch.nn
from torchsummary import summary

from model.vae import *

if __name__ == '__main__':
    dimensions = [784, 500, 500, 2000, 10]
    model_params = {
        'decoder_final_activation': 'sigmoid',
    }

    vae = VAE(dimensions, **model_params)
    summary(vae, (784,))