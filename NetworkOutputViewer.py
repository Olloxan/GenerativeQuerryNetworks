import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.distributions import Normal
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from torchvision.utils import make_grid

from shepardmetzler import ShepardMetzler
from gqn import GenerativeQueryNetwork, partition

cuda = False #torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

plt.ion()

image = np.zeros((64, 64,3))
fig, ax = plt.subplots(1,2, figsize=(15,10))
ax[0].set_xlabel("input")
img1 = ax[0].imshow(image)
ax[1].set_xlabel("prediction")
img2 = ax[1].imshow(image)

def displayImage(image1, image2):              
    ax[0].set_title("Hello")
    ax[0].imshow(image1)
    ax[1].imshow(image2)    
    #fig.canvas.draw()
    plt.show()
    plt.pause(1.0)
       
# model path
modelName = "gqn_model_rooms_n_ep48_49perc"
#modelPath = "D:\\Projekte\\MachineLearning\\generative-query-network-pytorch\\model\\"
modelPath = "Y:\\"
modelFullPath = modelPath + modelName

    

model = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=8)
pretrained_dict = torch.load(modelFullPath, map_location='cpu')#.to(device)
model.load_state_dict(pretrained_dict)
model = model.to(device)

# datapath
#datapath = "D:\\Projekte\\MachineLearning\\DataSets\\shepard_metzler_7_parts"
#datapath = "D:\\Projekte\\MachineLearning\\DataSets\\temp"
datapath = "Z:\\Datasets\\rooms_free_camera_no_object_rotations\\rooms_free_camera_no_object_rotations\\temp"
valid_dataset = ShepardMetzler(root_dir=datapath, fraction=1.0, train=False)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

valid_imgs, valid_viewpoints = next(iter(valid_loader))
part_valid_imgs, part_valid_viewpoints, part_context_imgs, part_context_viewpoints = partition(valid_imgs, valid_viewpoints)

model.eval()

with torch.no_grad():
    
    #prediction, ground_truth, _ = model(part_valid_imgs, part_valid_viewpoints, part_context_imgs, part_context_viewpoints)
    
    for valid_imgs, viewpoints, context_img, context_viewpoint in zip(part_valid_imgs, part_valid_viewpoints, part_context_imgs, part_context_viewpoints):    
        # Reconstruction, representation and divergence
        prediction, _, _ = model(valid_imgs.unsqueeze(0), viewpoints.unsqueeze(0), context_img.unsqueeze(0), context_viewpoint.unsqueeze(0))    
        
        viewpoint = context_viewpoint.numpy()

        # Send to CPU
        prediction = prediction.detach().cpu().float()
        outputImage = prediction.squeeze(0).numpy().transpose((1,2,0))
        
        # reconstruction
        inputImages = make_grid(valid_imgs, nrow=4).numpy().transpose((1,2,0))
        displayImage(inputImages, outputImage)
       
            

    

    

    
    
   
    

    

        
    