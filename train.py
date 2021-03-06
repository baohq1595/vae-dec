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
from dataloader.metagenomics_dataset import GenomeDataset
from transform.gene_transforms import numerize_genome_str

pretrained_save_path = 'model/pretrained/model.pt'
def pretrain(model: ClusteringBasedVAE, train_dataloader, val_dataloader, **params):
    if os.path.exists(pretrained_save_path):
        model.load_state_dict(torch.load(pretrained_save_path))
        return
    else:
        os.makedirs(os.path.dirname(pretrained_save_path))

    num_pretrained_epoch = params.get('pretrained_epochs', 10)
    save_path = params.get('save_path', 'output/model')
    dataset_name = params.get('dataset_name', '')
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    model = model.to(device)

    res_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(itertools.chain(model.encoder.parameters(),
                                model.decoder.parameters()), lr=0.002)
    steplr = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

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
            
            x = x.float()
            x = x.to(device)
            # Forward pass
            _, z_mu, _ = model.encoder(x)
            x_decoded = model.decoder(z_mu)
            loss = res_loss(x_decoded, x)
            total_loss += loss.detach().cpu().numpy()

            # Calculate gradient and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters += 1
        
        print('VAE resconstruction loss: ', total_loss / iters)
        steplr.step()
    
    model.encoder.sampling.log_var.load_state_dict(model.encoder.sampling.mu.state_dict())

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
        total_loss = 0.0
        for i, data in enumerate(train_dataloader):
            steplr.step()
            model.zero_grad()

            # Get only data, ignore label (data[1])
            x = data[0]

            # Flatten 28x28 to 1x784 on mnist dataset
            if dataset_name == 'mnist':
                x = x.view(x.size()[0], -1)
            
            x = x.to(model.device)

            # Acquire the loss
            loss = model.elbo_loss(x, 1)

            # Calculate gradients
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            # Update models
            optimizer.step()

            train_iters += 1

            total_loss += loss.detach().cpu().numpy()

        print('Training loss: ', total_loss / train_iters)


        gtruth = []
        predicted = []
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

                gtruth.append(labels)

                # Cluster latent space
                sample = np.argmax(gamma.cpu().detach().numpy(), axis=1)
                predicted.append(sample)
                # mean_accuracy += cluster_accuracy(sample, labels)[0]
                iters += 1

            gtruth = np.concatenate(gtruth, 0)
            predicted = np.concatenate(predicted, 0)
            print('accuracy p(c|z): {:0.4f}'.format(cluster_accuracy(predicted,gtruth)[0]*100))

    torch.save(model.models.state_dict(), os.path.join(save_path, 'vae-dec-model-{}'
        .format(strftime("%Y-%m-%d-%H-%M", gmtime())
    )))

    # plt.show()


def get_metagenomics_dataloader():
    gen_transforms = transforms.Compose([
        numerize_genome_str,
        transforms.ToTensor()
    ])
    genomics_dataset = GenomeDataset('data/gene/L1.fna', transform=numerize_genome_str)
    dataloader = DataLoader(genomics_dataset, batch_size=32, shuffle=True)

    return dataloader

if __name__ == '__main__':
    dimensions = [400, 500, 500, 2000, 10]
    model_params = {
        'decoder_final_activation': 'relu',
        'pretrained_epochs': 1,
        'epochs': 1,
        'save_path': 'output/model',
        'dataset_name': 'mnist'
    }

    dec_cluster = ClusteringBasedVAE(2, dimensions, 1, **model_params)

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

    gen_dataloader = get_metagenomics_dataloader()

    if torch.cuda.is_available():
        print('Cuda is available')
        dec_cluster = dec_cluster.cuda()    
    else:
        print('No GPU')
    pretrain(dec_cluster, gen_dataloader, gen_dataloader, **model_params)
    train(dec_cluster, gen_dataloader, gen_dataloader, **model_params)