"""
run-gqn.py

Script to train the a GQN on the Shepard-Metzler dataset
in accordance to the hyperparameter settings described in
the supplementary materials of the paper.
"""
import random
import math
from argparse import ArgumentParser

# Torch
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from ignite.metrics import RunningAverage

from gqn import GenerativeQueryNetwork, partition, Annealer
from shepardmetzler import ShepardMetzler
#from placeholder import PlaceholderData as ShepardMetzler

from logger import Logger

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

# Random seeding
random.seed(99)
torch.manual_seed(99)
if cuda: torch.cuda.manual_seed(99)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger = Logger()

datapath = "D:\\Projekte\\MachineLearning\\DataSets\\temp"

if __name__ == '__main__':
    parser = ArgumentParser(description='Generative Query Network on Shepard Metzler Example')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs run (default: 200)')
    parser.add_argument('--batch_size', type=int, default=1, help='multiple of batch size (default: 1)')
    parser.add_argument('--data_dir', type=str, help='location of data', default="D:\\Machine Learning\\Datasets\\shepard_metzler_5_parts")
    parser.add_argument('--log_dir', type=str, help='location of logging', default="log")
    parser.add_argument('--fraction', type=float, help='how much of the data to use', default=1.0)
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--data_parallel', type=bool, help='whether to parallelise based on data (default: False)', default=False)
    args = parser.parse_args()

    
    # Create model and optimizer
    model = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=8).to(device)    
    #pretrained_dict = torch.load("model/gqn_model_cp_ep5_73perc", map_location='cpu')#.to(device)
    #model.load_state_dict(pretrained_dict)
    model = nn.DataParallel(model) if args.data_parallel else model
    

    optimizer = torch.optim.Adam(model.parameters(), lr=5 * 10 ** (-5))

    # Rate annealing schemes
    sigma_scheme = Annealer(2.0, 0.7, 80000)
    mu_scheme = Annealer(5 * 10 ** (-6), 5 * 10 ** (-6), 1.6 * 10 ** 5)

    # Load the dataset
    train_dataset = ShepardMetzler(root_dir=datapath, fraction=args.fraction)
    valid_dataset = ShepardMetzler(root_dir=datapath, fraction=args.fraction, train=False)

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    

    imgs, viewpoints = next(iter(train_loader))
    batch = next(iter(train_loader))
    print("Batch Tensor: ")    
    print(batch[0].shape)
    print(batch[1].shape)
    #x, v = x.to(device), v.to(device)
    #x, v, x_q, v_q = partition(x, v)
    print("img Batch Tensor: ")    
    print(imgs.shape)
    print("viewpoints Batch Tensor: ")
    print(viewpoints.shape)
    
   
    #def step(batch):
    model.train()

    x, v = batch
    x, v = x.to(device), v.to(device)
    x, v, x_q, v_q = partition(x, v)

    # Reconstruction, representation and divergence
    x_mu, _, kl = model(x, v, x_q, v_q)
       
    # Log likelihood
    sigma = next(sigma_scheme)
    ll = Normal(x_mu, sigma).log_prob(x_q)

    likelihood     = torch.mean(torch.sum(ll, dim=[1, 2, 3]))
    kl_divergence  = torch.mean(torch.sum(kl, dim=[1, 2, 3]))

    # Evidence lower bound
    elbo = likelihood - kl_divergence
    loss = -elbo
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()         
    
      
    def save_images(engine):
        print("Epoch Completed save_images")
        with torch.no_grad():
            x, v = engine.state.batch
            x, v = x.to(device), v.to(device)
            x, v, x_q, v_q = partition(x, v)

            x_mu, r, _ = model(x, v, x_q, v_q)

            r = r.view(-1, 1, 16, 16)

            # Send to CPU
            x_mu = x_mu.detach().cpu().float()
            r = r.detach().cpu().float()

            writer.add_image("representation", make_grid(r), engine.state.epoch)
            writer.add_image("reconstruction", make_grid(x_mu), engine.state.epoch)

   
    def validate(engine):
        print("Epoch Completed validate")
        model.eval()
        with torch.no_grad():
            x, v = next(iter(valid_loader))
            x, v = x.to(device), v.to(device)
            x, v, x_q, v_q = partition(x, v)

            # Reconstruction, representation and divergence
            x_mu, _, kl = model(x, v, x_q, v_q)

            # Validate at last sigma
            ll = Normal(x_mu, sigma_scheme.recent).log_prob(x_q)

            likelihood = torch.mean(torch.sum(ll, dim=[1, 2, 3]))
            kl_divergence = torch.mean(torch.sum(kl, dim=[1, 2, 3]))

            # Evidence lower bound
            elbo = likelihood - kl_divergence

            writer.add_scalar("validation/elbo", elbo.item(), engine.state.epoch)
            writer.add_scalar("validation/kl", kl_divergence.item(), engine.state.epoch)

      

