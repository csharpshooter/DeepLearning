import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import cv2

import src.utils.utils as utils
from src.utils.modelutils import ModelUtils


class GradcamExperiment:

    def get_all_layers(net):

        def hook_fn(m, i, o):
            visualisation[m] = o
            output.append(o)
            names.append(m)

        visualisation = {}
        output = []
        names = []

        for name, layer in net._modules.items():
            # If it is a sequential, don't register a hook on it
            # but recursively register hook on all it's module children
            if isinstance(layer, nn.Sequential):
                GradcamExperiment.get_all_layers(layer)
            else:
                # it's a non sequential. Register a hook
                layer.register_forward_hook(hook_fn)

        return visualisation, output, names

    def test(self):
        image_paths = glob.glob('./images/testimages/*.*')
        images = list(map(lambda x: Image.open(x), image_paths))
        ModelUtils.subplot(images, title='inputs', nrows=2, ncols=5)

        inputs = [
            torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                             std=(0.5, 0.5, 0.5))])(
                x).unsqueeze(0) for x in images]  # add 1 dim for batch

        print(torch.cuda.is_available())
        saved_data, epoch, model_state_dict, optimizer_state_dict, train_losses, train_acc, test_losses, test_acc \
            , test_losses, lr_data, class_correct, class_total \
            = utils.Utils.loadmodel(path="savedmodels/finalmodelwithdata.pt")

        model, device = utils.Utils.createmodelresnet18(model_state_dict=model_state_dict)

        GradcamExperiment.get_all_layers(model)

        inputs = [i.to(device) for i in inputs]

        pred = model(inputs[0])

        class_pred = int(np.array(pred.cpu().argmax(dim=1)))
        utils.Utils.imshow(inputs[0][0].cpu())

        heatmap = GradcamExperiment.getheatmap(pred, class_pred, model, inputs[0][0].cpu())
        plt.imshow(heatmap.squeeze())

        superimposed = GradcamExperiment.superposeimage(heatmap, inputs[0][0])

        plt.imshow(superimposed, cmap='gray', interpolation='bicubic')

    def getheatmap(pred, class_pred, netx, img):
        # get the gradient of the output with respect to the parameters of the model
        pred[:, class_pred].backward()
        # pull the gradients out of the model
        gradients = netx.get_activations_gradient()

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = netx.get_activations(img.cuda()).detach()

        # weight the channels by corresponding gradients
        for i in range(512):
            activations[:, i, :, :] *= pooled_gradients[i]

        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap.cpu(), 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)
        # heatmap = None
        return heatmap

    def superposeimage(heatmap, img):
        heat1 = np.array(heatmap)
        heatmap1 = cv2.resize(heat1, (img.shape[1], img.shape[0]))
        heatmap1 = np.uint8(255 * heatmap1)
        heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
        superimposed_img = heatmap1 * 0.4 + img
        return superimposed_img;
