import time
import math
from argparse import ArgumentParser

import torch
import torch.nn as nn

from torch.distributions import Normal
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from shepardmetzler import ShepardMetzler

from gqn import GenerativeQueryNetwork, partition, Annealer
from logger import Logger
from myTimer import myTimer

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

def step(batch, iteration):
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

        with torch.no_grad():
            # Anneal learning rate
            mu = next(mu_scheme)
            i = iteration
            for group in optimizer.param_groups:
                group["lr"] = mu * math.sqrt(1 - 0.999 ** i) / (1 - 0.9 ** i)

        return {"elbo": elbo.item(), "kl": kl_divergence.item(), "sigma": sigma, "mu": mu}

if __name__ == '__main__':
    logger = Logger()
    timer = myTimer()
    parser = ArgumentParser(description='Generative Query Network on Shepard Metzler Example')
    parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs run (default: 200)')
    parser.add_argument('--batch_size', type=int, default=1, help='multiple of batch size (default: 1)')
    parser.add_argument('--data_dir', type=str, help='location of data', default="D:\\Machine Learning\\Datasets\\shepard_metzler_5_parts")
    parser.add_argument('--log_dir', type=str, help='location of logging', default="log")
    parser.add_argument('--fraction', type=float, help='how much of the data to use', default=1.0)
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--data_parallel', type=bool, help='whether to parallelise based on data (default: False)', default=False)
    args = parser.parse_args()

    # Create model and optimizer
    model = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=8).to(device)
    model = nn.DataParallel(model) if args.data_parallel else model
    torch.save(model.state_dict(), "model/gqn_model")
    optimizer = torch.optim.Adam(model.parameters(), lr=5 * 10 ** (-5))

    # Rate annealing schemes
    sigma_scheme = Annealer(2.0, 0.7, 80000)
    mu_scheme = Annealer(5 * 10 ** (-6), 5 * 10 ** (-6), 1.6 * 10 ** 5)

    # Load the dataset
    train_dataset = ShepardMetzler(root_dir=args.data_dir, fraction=args.fraction)
    valid_dataset = ShepardMetzler(root_dir=args.data_dir, fraction=args.fraction, train=False)

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    dataloader_Iter = iter(train_loader)
    iteration = 0
    epoch_length = len(train_loader)
    
    switch_variable = 0
    timer.update(time.time())

    for epoch in range(args.n_epochs):        
        while iteration < epoch_length:

            batch = next(dataloader_Iter)
            iteration += 1
            output = step(batch, iteration)
        
            timer.update(time.time())
            timediff = timer.getTimeDiff()
            total_time = timer.getTotalTime()

            loopstogo = (epoch_length - iteration)
            estimatedtimetogo = timer.getTimeToGo(loopstogo)
            logger.printDayFormat("runntime last epochs: ", timediff)
            logger.printDayFormat("total runtime: ", total_time)
            logger.printDayFormat("estimated time to run: ", estimatedtimetogo)          

            if iteration % 50 == 0:            
                logger.log_state_dict(model.state_dict(), "model/gqn_model_{0}".format(switch_variable))
                switch_variable += 1
                switch_variable %= 2

    logger.log_state_dict(model.state_dict(), "model/gqn_model")