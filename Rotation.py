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

pi = 3.1415629

plt.ion()

image = np.zeros((64, 64,3))
fig, ax = plt.subplots(1,2, figsize=(15,10))
ax[0].set_xlabel("input")
img1 = ax[0].imshow(image)
ax[1].set_xlabel("prediction")
img2 = ax[1].imshow(image)

def displayImage(image1, image2, yaw):              
    ax[0].set_title("Inputsequence")
    ax[0].imshow(image1)
    ax[1].set_title(r"Yaw: " + str(yaw))
    ax[1].imshow(image2)    
    
    plt.show()
    plt.pause(0.01)
       
# model path
modelName = "model-checkpoint.pth"
modelPath = "D:\\Projekte\\MachineLearning\\generative-query-network-pytorch\\model\\"
modelFullPath = modelPath + modelName

model_trained = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=8)
pretrained_dict = torch.load(modelFullPath, map_location='cpu')#.to(device)
model_trained.load_state_dict(pretrained_dict)
model_trained = model_trained.to(device)

# datapath
datapath = "D:\\Projekte\\MachineLearning\\DataSets\\shepard_metzler_7_parts"
valid_dataset = ShepardMetzler(root_dir=datapath, fraction=1.0, train=False)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

valid_imgs, valid_viewpoints = next(iter(valid_loader))
part_valid_imgs, part_valid_viewpoints, part_context_imgs, part_context_viewpoints = partition(valid_imgs, valid_viewpoints)

batch_size, num_views, channels, height, width = part_valid_imgs.shape

model_trained.eval()

with torch.no_grad():        
    
    for valid_imgs, viewpoints, context_img, context_viewpoint in zip(part_valid_imgs, part_valid_viewpoints, part_context_imgs, part_context_viewpoints):    
        # Reconstruction, representation and divergence        

        #x_ = valid_imgs.view(-1, channels, height, width)
        #v_ = viewpoints.view(-1, 7)
        
        
        phi_trained = model_trained.representation(valid_imgs, viewpoints)    
        rep_trained = torch.sum(phi_trained, dim=0)
        
        inputImages = make_grid(valid_imgs, nrow=4).numpy().transpose((1,2,0))

        for i in range(8):
            rotatingViewPoint = torch.zeros(7).copy_(context_viewpoint)
            
            yaw = (i+1) * (pi/8) - pi/2
            rotatingViewPoint[3], rotatingViewPoint[4] = np.cos(yaw), np.sin(yaw)

            prediction = model_trained.generator.sample((height, width), rotatingViewPoint.unsqueeze(0), rep_trained)


            # Send to CPU
            prediction = prediction.detach().cpu().float()
            outputImage = prediction.squeeze(0).numpy().transpose((1,2,0))
        
            # reconstruction
        
            displayImage(inputImages, outputImage, yaw)
       
            

    

    

    
    
   
    

    

        
    
