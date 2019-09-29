import torch
import torch.nn as nn
from torch.nn import init

from utils.distributions import log_gaussian, log_standard_gaussian

activations_mapping = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'linear': None
}

class VAE(nn.Module):
    '''
    Variational autoencoder module for deep embedding clustering.
    Args:
        dimensions: list of layers' dimensions for each block encoder/decoder
        For ex: [784, 500, 500, 2000, 10], encoder will have input layer dim 784,
        500-500-2000 is hidden layers' dimension, 10 is latent's dimension.
        Decoder will have reversed order, which is [10, 2000, 500, 500, 784].
        **kwargs: other params dict, contains:
        kwargs['decoder_final_activation'] is final activation function for decoder last
        layer (ex. mnist data use sigmoid, reuters data uses linear,...).
    '''
    def __init__(self, dimensions, **kwargs):
        super(VAE, self).__init__()
        assert len(dimensions) > 1

        # unpack dimension of vae
        self.embedding_dim = dimensions[0]
        self.hidden_dims = dimensions[1:-1]
        self.latent_dim = dimensions[-1]
        self.dec_final_act = kwargs['decoder_final_activation']
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        self.is_logits = kwargs.get('logits', False)

        if self.is_logits:
            self.resconstruction_loss = nn.modules.loss.MSELoss()
        else:
            self.resconstruction_loss = self.binary_cross_entropy

        # Construct layers for encoder and decoder block
        # Encoder block
        self.enc_hidden_layers = nn.Sequential()
        self.enc_hidden_layers.add_module('hidden_layer_0',
            nn.Linear(self.embedding_dim, self.hidden_dims[0]))
        self.enc_hidden_layers.add_module('h_layer_act_0', nn.ReLU())
        
        for i, _ in enumerate(self.hidden_dims[:-1]):
            self.enc_hidden_layers.add_module('hidden_layer_{}'.format(i + 1),
                nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1])),
            self.enc_hidden_layers.add_module('h_layer_act_{}'.format(i + 1), nn.ReLU())
        
        # define mean and log variance of vae
        self.z_mean = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.z_log_var = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        # ~Encoder block

        # Decoder block
        dec_hidden_layers = nn.Sequential()
        dec_hidden_layers.add_module('hidden_layer_0',
            nn.Linear(self.latent_dim, self.hidden_dims[-1]))
        dec_hidden_layers.add_module('h_layer_act_0',
            nn.ReLU())
        reversed_hidden_dims = list(reversed(self.hidden_dims))
        for i, _ in enumerate(reversed(self.hidden_dims)):
            if i == (len(reversed_hidden_dims) - 1):
                dec_hidden_layers.add_module('hidden_layer_{}'.format(i + 1),
                    nn.Linear(reversed_hidden_dims[i], self.embedding_dim)),
            else:
                dec_hidden_layers.add_module('hidden_layer_{}'.format(i + 1),
                    nn.Linear(reversed_hidden_dims[i], reversed_hidden_dims[i + 1])),
                dec_hidden_layers.add_module('h_layer_act_{}'.format(i + 1), nn.ReLU())
        
        # Final activation function of decoder depends on data
        if self.dec_final_act is not None:
            if self.dec_final_act == 'sigmoid':
                dec_hidden_layers.add_module('dec_final_act', nn.Sigmoid())
            elif self.dec_final_act == 'tanh':
                dec_hidden_layers.add_module('dec_final_act', nn.Tanh())
            elif self.dec_final_act == 'relu':
                dec_hidden_layers.add_module('dec_final_act', nn.ReLU())
            else:
                pass
        self.decoder = dec_hidden_layers
        # ~ Decoder block

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.to(self.device)
    
    def binary_cross_entropy(self, r, x):
        return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)
    
    def encode(self, x):
        '''
        Encoding step. Receives data input tensor and produce is gaussian distribution
        params, which is mean and log-variance.
        Args:
            x: data input
        '''
        hidden_embedding = self.enc_hidden_layers(x)
        z_mean = self.z_mean(hidden_embedding)
        z_log_var = self.z_log_var(hidden_embedding)

        latent = self.sampling(z_mean, z_log_var)

        return latent, z_mean, z_log_var
    
    def decode(self, z):
        '''
        Decoding step. Receives latent vector encoded from encoder, and decode
        it back to observed data distribution.
        Args:
            z: latent vector.
        '''
        return self.decoder(z)


    def sampling(self, z_mean, z_log_var):
        '''
        Sampling function, which produce latent vector from params mean
        and log variance using formula as:
            z = z_mean + z_var*epsilon
        Args:
            z_mean: mean value of a gaussian.
            z_log_var: log of variance of a gaussian.
        '''
        epsilon = torch.randn(z_mean.size()).to(self.device)
        z = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z

    def _kld(self, z, q_param, p_param=None):
        """
        Computes the KL-divergence of
        some element z.

        KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                 = -E[log p(z) - log q(z)]

        :param z: sample from q-distribuion
        :param q_param: (mu, log_var) of the q-distribution
        :param p_param: (mu, log_var) of the p-distribution
        :return: KL(q||p)
        """
        (q_mu, q_log_var) = q_param
        qz = log_gaussian(z, q_mu, q_log_var)

        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            (p_mu, p_log_var) = p_param
            pz = log_gaussian(z, p_mu, p_log_var)

        kl = qz - pz

        return kl

    def criterion(self, x, x_decoded, z, z_mean, z_log_var):
        return - self.resconstruction_loss(x_decoded, x) -\
            self._kld(z, (z_mean, z_log_var))

    
    def forward(self, x):
        latent, z_mean, z_log_var = self.encode(x)
        x_hat = self.decode(latent)

        return x_hat, latent, z_mean, z_log_var
