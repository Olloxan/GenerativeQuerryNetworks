import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from shepardmetzler import ShepardMetzler
from gqn import partition

plt.ion()

image = np.zeros((64, 64,3))
fig, ax = plt.subplots()
im = ax.imshow(image)
    

def displayImage(image, step=1, reward=1):              
    title = "step: {0} reward: {1:.2f}".format(step, reward)
    plt.title(title)        
    im.set_data(image)
    fig.canvas.draw()
    plt.pause(0.05)

datapath = "D:\\Projekte\\MachineLearning\\DataSets\\shepard_metzler_7_parts"

train_dataset = ShepardMetzler(root_dir=datapath, fraction=1)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)


valid_dataset = ShepardMetzler(root_dir=datapath, fraction=1.0, train=False)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

train_imgs, train_viewpoints = next(iter(train_loader))
valid_imgs, valid_viewpoints = next(iter(valid_loader))

part_train_imgs, part_train_viewpoints, context_imgs, context_viewpoints = partition(train_imgs, train_viewpoints)

for parts in part_train_imgs:    
    for img in parts:
        showimg = img.numpy().transpose((1,2,0))
        displayImage(showimg)
        

 #test = representation numpy().transpose((1,2,0))

x=5