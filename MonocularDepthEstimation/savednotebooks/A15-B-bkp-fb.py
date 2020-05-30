#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/csharpshooter/EVA/blob/master/A12/A12-A/A12-A.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Import Libraries

# In[1]:


import datetime

from torch import nn

from src.dataset.monocularhelper import MonocularHelper
from src.train.torchvision import collate_fn

from src.train.torchvision.engine import train_one_epoch, evaluate

from src.models import get_instance_segmentation_model
import torchvision

print("Model execution started at:" + datetime.datetime.today().ctime())

from src.imports import *

import torch

print(torch.__version__)

# In[25]:


batch_size = 64
helper = MonocularHelper

final_output = r'/media/abhijit/DATA/Development/TSAI/EVA/MaskRCNN Dataset/OverLayedImages'
final_output_mask = r'/media/abhijit/DATA/Development/TSAI/EVA/MaskRCNN Dataset/OverLayedMask'
final_output_dm = r'/media/abhijit/DATA/Development/TSAI/EVA/MaskRCNN Dataset/OverLayedDepthMasks'
bg_path = r'/media/abhijit/DATA/Development/TSAI/EVA/MaskRCNN Dataset/Background'

# path = helper.download_dataset(folder_path="data")
# classes, id_dict = helper.get_classes(path)

train_data, train_label, test_data, test_label = helper.get_train_test_data(final_output_mask, final_output,
                                                                            final_output_dm, 40, 400000, bg_path)

mean_timgnet = [0.48043722, 0.44820285, 0.39760238]
std_timgnet = [0.27698976, 0.26908714, 0.2821603]

train_transforms, test_transforms = preprochelper.PreprocHelper.getpytorchtransforms(mean_timgnet, std_timgnet)
ds = dst.Dataset()

train_dataset = ds.get_monocular_train_dataset(train_image_data=train_data, train_image_labels=train_label,
                                               train_transforms=train_transforms)

test_dataset = ds.get_monocular_test_dataset(test_image_labels=test_label, test_image_data=test_data,
                                             test_transforms=test_transforms)

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(train_dataset)).tolist()
dataset = torch.utils.data.Subset(train_dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(test_dataset, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=collate_fn)

img_list = utils.Utils.loadfiles(train_data, test_data)

mean, std = utils.Utils.calculate_mean_std_deviation(img_list, None)

# mean,std = utils.Utils.calculate_mean_std_deviation(data_loader)

# data_iterator = iter(data_loader)

# plotdata.PlotData.showImagesfromdataset(data_iterator, values=None, image_count=20, col=5)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2

# get the model using our helper function
# model = get_instance_segmentation_model(num_classes)
# move model to the right device

model = torchvision.models.resnet50(pretrained=True)

# for name, param in model.named_parameters():
#     if "weight" in name:
#         nn.init.kaiming_normal_(param, nonlinearity="relu")
#     # elif "bias" in name:
#     #     nn.init.constant_(param, 0)

model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# let's train it for 10 epochs
num_epochs = 10

train_model = train.TrainModel()

# torch.autograd.set_detect_anomaly(True)
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_model.train_Monocular(model, device, data_loader, optimizer, epoch)
    # train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)
