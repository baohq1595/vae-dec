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

def pretrain(model: ClusteringBasedVAE, train_dataloader, val_dataloader, **params):
    num_pretrained_epoch = params.get('pretrained_epochs', 10)
    save_path = params.get('save_path', 'output/model')
    dataset_name = params.get('dataset_name', '')
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    model = model.to(device)

    res_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(itertools.chain(model.vae.encoder.parameters(),
                                model.vae.decoder.parameters()))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('Pretrains VAE using only reconstruction loss...')
    for pre_epoch in range(num_pretrained_epoch):
        total_loss = 0.0
        iters = 0
        for i, data in enumerate(train_dataloader):
            x = data[0]

            # Flatten 28x28 to 1x784 on mnist dataset
            if dataset_name == 'mnist':
                x = x.view(x.size()[0], -1)
            
            x = x.to(device)
            # Forward pass
            x_decoded, _, _, _ = model.vae(x)
            loss = res_loss(x_decoded, x)
            total_loss = loss.detach().cpu().numpy()

            # Calculate gradient and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters += 1
        
        print('VAE resconstruction loss: ', total_loss / (len(train_dataloader) / iters))
    
    model.vae.encoder.sampling.log_var.load_state_dict(model.vae.encoder.sampling.mu.state_dict())

    Z = []
    Y = []
    with torch.no_grad():
        for x, y in train_dataloader:
            # Flatten 28x28 to 1x784 on mnist dataset
            if dataset_name == 'mnist':
                x = x.view(x.size()[0], -1)
            x = x.to(device)
            z, mu, log_var = model.vae.encoder(x)
            assert F.mse_loss(mu, log_var) == 0
            Z.append(mu)
            Y.append(y)

    Z = torch.cat(Z, 0).detach().cpu().numpy()
    Y = torch.cat(Y, 0).detach().numpy()
    gmm = GaussianMixture(n_components=model.n_centroids, covariance_type='diag')
    predict = gmm.fit_predict(Z)

    print('Accuracy = {:.4f}%'.format(cluster_accuracy(predict, Y)[0] * 100))

    model.mu_param.data = torch.from_numpy(gmm.means_).to(device).float()
    model.lambda_param.data = torch.log(torch.from_numpy(gmm.covariances_).to(device).float())
    model.theta_param.data = torch.from_numpy(gmm.weights_).to(device).float()

    # torch.save(model.state_dict(), 'results/model/pretrained/pretrained_model.pk')


def train(model, train_dataloader, val_dataloader, **params):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, eps=1e-4)
    steplr = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.9)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epochs = params.get('epochs', 10)
    num_pretrained_epoch = params.get('pretrained_epochs', 10)
    save_path = params.get('save_path', 'output/model')
    dataset_name = params.get('dataset_name', '')



    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(num_epochs):
        train_iters = 0
        for i, data in enumerate(train_dataloader):
            steplr.step()
            model.zero_grad()

            # Get only data, ignore label (data[1])
            x = data[0]

            # Flatten 28x28 to 1x784 on mnist dataset
            if dataset_name == 'mnist':
                x = x.view(x.size()[0], -1)
            
            x = x.to(model.device)

            # Forward thru vae model

            # Acquire the loss
            loss = model.elbo_loss(x, 1)

            # Calculate gradients
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            # Update models
            optimizer.step()

            train_iters += 1

            if train_iters % 100 == 0:
                print('Training loss: ', loss.detach().cpu().numpy())

        # For each epoch, log the p_c_z accuracy
        with torch.no_grad():
            mean_accuracy = 0.0
            iters = 0
            for i, data in enumerate(val_dataloader):
                # Get z value
                x = data[0].to(model.device)
                labels = data[1].cpu().detach().numpy()
                if dataset_name == 'mnist':
                    x = x.view(x.size()[0], -1)
                
                # x_decoded, latent, z_mean, z_log_var, gamma = model(x)
                gamma = model(x)

                # Cluster latent space
                sample = np.argmax(gamma.cpu().detach().numpy(), axis=1)
                mean_accuracy += cluster_accuracy(sample, labels)[0]
                iters += 1

            print('accuracy p(c|z): %0.8f' % (mean_accuracy / iters))

    torch.save(model.models.state_dict(), os.path.join(save_path, 'vae-dec-model-{}'
        .format(strftime("%Y-%m-%d-%H-%M", gmtime())
    )))

    # plt.show()

if __name__ == '__main__':
    dimensions = [784, 500, 500, 2000, 10]
    model_params = {
        'decoder_final_activation': 'sigmoid',
        'pretrained_epochs': 50,
        'epochs': 200,
        'save_path': 'output/model',
        'dataset_name': 'mnist'
    }

    dec_cluster = ClusteringBasedVAE(10, dimensions, 1, **model_params)

    train_dataloader = DataLoader(MNIST('data/mnist', train=True, 
                                        download=True,
                                        transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        # transforms.Normalize((0.1307,), (0.3081,)),
                                                        # transforms.Normalize((0,), (1,))
                                                ])),
                                batch_size=32,
                                shuffle=True)

    val_dataloader = DataLoader(MNIST('data/mnist', train=False, 
                                        download=True,
                                        transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        # transforms.Normalize((0.1307,), (0.3081,)),
                                                        # transforms.Normalize((0,), (1,))
                                                ])),
                                batch_size=32,
                                shuffle=True)

    if torch.cuda.is_available():
        print('Cuda is available')
        dec_cluster = dec_cluster.cuda()    
    else:
        print('No GPU')
    pretrain(dec_cluster, train_dataloader, val_dataloader, **model_params)
    train(dec_cluster, train_dataloader, val_dataloader, **model_params)