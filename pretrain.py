import torch
import torch.nn
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision import transforms

from model.vae import *
from model.dec import *
from utils.utils import *
import matplotlib.pyplot as plt

def pretrain(model, train_dataloader, val_dataloader, **params):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, eps=1e-4)
    num_epochs = params.get('epochs', 10)
    save_path = params.get('save_path', 'output/model')
    dataset_name = params.get('dataset_name', '')
    plt.show()
    is_first = True

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(num_epochs):
        for i, data in enumerate(train_dataloader):
            model.zero_grad()

            # Get only data, ignore label (data[1])
            x = data[0]

            # Flatten 28x28 to 1x784 on mnist dataset
            if dataset_name == 'mnist':
                x = x.view(x.size()[0], -1)
            
            x = x.to(model.device)

            # Forward thru vae model
            x_decoded, latent, z_mean, z_log_var = model(x)

            # Acquire the loss
            loss = model.criterion(x, x_decoded, latent, z_mean, z_log_var)
            loss = -torch.mean(loss)

            # Calculate gradients
            loss.backward()

            plot_grad_flow_lines(model.named_parameters())

            # Update models
            optimizer.step()

            plt.show()

        # For each epoch, log the p_c_z accuracy
        with torch.no_grad():
            val_loss = 0.0
            iters = 0
            for i, data in enumerate(val_dataloader):
                # Get z value
                x = data[0].to(model.device)
                labels = data[1].cpu().detach().numpy()
                if dataset_name == 'mnist':
                    x = x.view(x.size()[0], -1)
                
                x_decoded, latent, z_mean, z_log_var = vae(x)

                val_loss += model.criterion(x, x_decoded, latent, z_mean, z_log_var)
                iters += 1

            print('accuracy p(c|z): %0.8f' % (val_loss / iters))

    torch.save(model.models.state_dict(), os.path.join(save_path, 'vae-dec-model-{}'
        .format(strftime("%Y-%m-%d-%H-%M", gmtime())
    )))

if __name__ == '__main__':
    dimensions = [784, 500, 500, 2000, 10]
    model_params = {
        'decoder_final_activation': 'sigmoid',
        'epochs': 100,
        'save_path': 'output/model',
        'dataset_name': 'mnist'
    }

    vae = VAE(dimensions, **model_params)

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

    pretrain(vae, train_dataloader, val_dataloader, **model_params)