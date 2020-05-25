import torch
from torch.utils.data import DataLoader
from shepardmetzler import ShepardMetzler
from torchvision.utils import make_grid

from gqn import GenerativeQueryNetwork, partition

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")


model = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=8)

kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}
valid_dataset = ShepardMetzler(root_dir=args.data_dir, fraction=args.fraction, train=False)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)


model.eval()
with torch.no_grad():
    x, v = next(iter(valid_loader))
    x, v = x.to(device), v.to(device)
    x, v, x_q, v_q = partition(x, v)

    # Reconstruction, representation and divergence
    x_mu, _, kl = model(x, v, x_q, v_q)

    r = r.view(-1, 1, 16, 16)

    # Validate at last sigma
    ll = Normal(x_mu, sigma_scheme.recent).log_prob(x_q)

    # Send to CPU
    x_mu = x_mu.detach().cpu().float()
    r = r.detach().cpu().float()

    writer.add_image("representation", make_grid(r), engine.state.epoch)
    writer.add_image("reconstruction", make_grid(x_mu), engine.state.epoch)    

    likelihood = torch.mean(torch.sum(ll, dim=[1, 2, 3]))
    kl_divergence = torch.mean(torch.sum(kl, dim=[1, 2, 3]))

    # Evidence lower bound
    elbo = likelihood - kl_divergence

    writer.add_scalar("validation/elbo", elbo.item(), engine.state.epoch)
    writer.add_scalar("validation/kl", kl_divergence.item(), engine.state.epoch)