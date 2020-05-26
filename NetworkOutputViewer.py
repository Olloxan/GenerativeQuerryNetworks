import torch
from torch.utils.data import DataLoader
from torch.distributions import Normal
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from torchvision.utils import make_grid

from shepardmetzler import ShepardMetzler
from gqn import GenerativeQueryNetwork, partition, Annealer

cuda = False #torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

if __name__ == '__main__':

    writer = SummaryWriter(log_dir="log")

    sigma_scheme = Annealer(2.0, 0.7, 80000)
    mu_scheme = Annealer(5 * 10 ** (-6), 5 * 10 ** (-6), 1.6 * 10 ** 5)

    # model path
    modelName = "gqn_model_cp_ep39_83perc_shepardMetzler_7_parts"
    modelPath = "D:\\Projekte\\MachineLearning\\generative-query-network-pytorch\\model\\"
    modelFullPath = modelPath + modelName

    # datapath
    datapath = "D:\\Projekte\\MachineLearning\\DataSets\\shepard_metzler_7_parts"

    model = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=8)
    pretrained_dict = torch.load(modelFullPath, map_location='cpu')#.to(device)
    model.load_state_dict(pretrained_dict)
    model = model.to(device)

    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    valid_dataset = ShepardMetzler(root_dir=datapath, fraction=1.0, train=False)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, **kwargs)


    model.eval()
    with torch.no_grad():
        x, v = next(iter(valid_loader))
        x, v = x.to(device), v.to(device)
        x, v, x_q, v_q = partition(x, v)

        # Reconstruction, representation and divergence
        prediction, ground_truth, _ = model(x, v, x_q, v_q)

        ground_truth = ground_truth.view(-1, 1, 16, 16)

        # Send to CPU
        prediction = prediction.detach().cpu().float()
        ground_truth = ground_truth.detach().cpu().float()
    
        # representation
        test = make_grid(ground_truth).numpy().transpose((1,2,0))
        x = 5
        # reconstruction
        test2 = make_grid(prediction).numpy().transpose((1,2,0))

        plt.imshow(test2)
        plt.title('Transformed Images')
        #f, axarr = plt.subplots(1, 1)
        #axarr[0].imshow(test)
        #axarr[0].set_title('Transformed Images')

        #axarr[1].imshow(test2)
        #axarr[1].set_title('Transformed Images')

        plt.show()

        
    