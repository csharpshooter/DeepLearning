import datetime
import os.path
from os import path
from zipfile import ZipFile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.optim as optim
# from albumentations.pytorch import ToTensor
import torchvision
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

import src.dataset.dataset as dst
from src.models import CNN_Model, ResNet18, A11CustomResnetModel, MonocularModel, DepthModel
from src.models.MaskAndDepthModel import MaskAndDepthModel


class Utils:

    # helper function to un-normalize and display an image
    @staticmethod
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        plt.imshow(np.transpose(img, (1, 2, 0)))
        # plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

    def imshowt(tensor):
        tensor = tensor.squeeze()
        if len(tensor.shape) > 2:
            tensor = tensor.permute(1, 2, 0)
        img = tensor.cpu().numpy()
        plt.imshow(img, cmap='gray')
        plt.show()

    def printdatetime(self):
        print("Model execution started at:" + datetime.datetime.today().ctime())

    # def printgpuinfo():
    #     gpu_info = "nvidia-smi"
    #     # gpu_info = '\n'.join(gpu_info)
    #     print(gpu_info)

    def savemodel(model, epoch, path, optimizer_state_dict=None, train_losses=None, train_acc=None, test_acc=None,
                  test_losses=None, lr_data=None, class_correct=None, class_total=None):
        # Prepare model model saving directory.
        # save_dir = os.path.join(os.getcwd(), 'saved_models')
        t = datetime.datetime.today()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_state_dict,
            'train_losses': train_losses,
            'train_acc': train_acc,
            'test_losses': test_losses,
            'test_acc': test_acc,
            'lr_data': lr_data,
            'class_correct': class_correct,
            'class_total': class_total
        }, path)

    def loadmodel(path):
        checkpoint = torch.load(path)
        epoch = checkpoint['epoch']
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        train_losses = checkpoint['train_losses']
        train_acc = checkpoint['train_acc']
        test_losses = checkpoint['test_losses']
        test_acc = checkpoint['test_acc']
        lr_data = checkpoint['lr_data']
        class_correct = checkpoint['class_correct']
        class_total = checkpoint['class_total']
        return checkpoint, epoch, model_state_dict, optimizer_state_dict, train_losses, train_acc, test_losses, test_acc \
            , test_losses, lr_data, class_correct, class_total

    def createmodel(model_state_dict=None):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print(device)
        model = CNN_Model().to(device)

        return model, device

    def createmodelresnet18(model_state_dict=None, numclasses=10):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print(device)
        model = ResNet18(numclasses).to(device)

        if model_state_dict != None:
            model.load_state_dict(state_dict=model_state_dict)

        return model, device

    def createA11CustomResnetModel(model_state_dict=None):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print(device)
        model = A11CustomResnetModel().to(device)

        if model_state_dict != None:
            model.load_state_dict(state_dict=model_state_dict)

        return model, device

    def createMonocularModel(model_state_dict=None):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print(device)
        model = MonocularModel().to(device)

        if model_state_dict != None:
            model.load_state_dict(state_dict=model_state_dict)

        return model, device

    def createDepthModel(model_state_dict=None):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print(device)
        model = DepthModel().to(device)

        if model_state_dict != None:
            model.load_state_dict(state_dict=model_state_dict)

        return model, device

    def createMaskAndDepthModel(model_state_dict=None):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print(device)
        model = MaskAndDepthModel().to(device)

        if model_state_dict != None:
            model.load_state_dict(state_dict=model_state_dict)

        return model, device

    def createoptimizer(model, lr=0.1, momentum=0.9, weight_decay=0, nesterov=False):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
                              nesterov=nesterov)

        return optimizer

    def createscheduler(optimizer, mode, factor, patience=5, verbose=True, threshold=0.01,
                        threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience,
                                                               verbose=verbose, threshold=threshold,
                                                               threshold_mode=threshold_mode,
                                                               cooldown=cooldown, min_lr=min_lr, eps=eps)

        return scheduler

    def createschedulersteplr(optimizer, step_size=15, gamma=0.1):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

        return scheduler

    def calculate_mean_std_deviation_cifar10(self=None):

        traindataset = datasets.CIFAR10('./data', train=True, download=True, transform=ToTensor())
        testdataset = datasets.CIFAR10('./data', train=False, download=True, transform=ToTensor())

        data = np.concatenate([traindataset.data, testdataset.data], axis=0)
        data = data.astype(np.float32) / 255.

        means = []
        stdevs = []

        for i in range(3):  # 3 channels
            pixels = data[:, :, :, i].ravel()
            means.append(np.mean(pixels))
            stdevs.append(np.std(pixels))

        return [means[0], means[1], means[2]], [stdevs[0], stdevs[1], stdevs[2]]

    def calculate_mean_std_deviation(train_data, test_data):

        data = np.concatenate([train_data, test_data], axis=0)
        data = data.astype(np.float32) / 255.

        means = []
        stdevs = []

        for i in range(3):  # 3 channels
            pixels = data[:, :, :, i].ravel()
            means.append(np.mean(pixels))
            stdevs.append(np.std(pixels))

        return [means[0], means[1], means[2]], [stdevs[0], stdevs[1], stdevs[2]]

    def calculate_mean_std_deviation(data):

        data = data.astype(np.float32) / 255.

        means = []
        stdevs = []

        for i in range(3):  # 3 channels
            pixels = data[:, :, :, i].ravel()
            means.append(np.mean(pixels))
            stdevs.append(np.std(pixels))

        return [means[0], means[1], means[2]], [stdevs[0], stdevs[1], stdevs[2]]

    def showaccuracyacrossclasses(class_correct, class_total):
        classes = dst.Dataset.getclassesinCIFAR10dataset()
        for i in range(10):
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    classes[i], 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))

    def processarray(array, epochs):

        indexval = int(len(array) / epochs)
        start = 0
        end = indexval
        final = []
        while len(array) > end:
            # print(start)
            # print(end)
            temp = array[start:end]
            val = sum(temp) / len(temp)
            final.append(val)
            start = end
            end += indexval

        return final

    def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    def download_file(folder_path, url):

        # get the file name
        file_name = url.split("/")[-1]
        folder_path = folder_path + "/" + file_name

        if path.exists(folder_path):
            print('File: {} already downloaded.'.format(file_name))
            return folder_path

        # read 1024 bytes every time
        buffer_size = 1024
        # download the body of response by chunk, not immediately
        response = requests.get(url, stream=True)

        # get the total file size
        file_size = int(response.headers.get("Content-Length", 0))

        # progress bar, changing the unit to bytes instead of iteration (default by tqdm)
        progress = tqdm(response.iter_content(buffer_size), f"Downloading {folder_path}", total=file_size, unit="B",
                        unit_scale=True, unit_divisor=1024)
        with open(folder_path, "wb") as f:
            for data in progress:
                # write data read to the file
                f.write(data)
                # update the progress bar manually
                progress.update(len(data))

        return folder_path

    def extract_zip_file(file_path, extract_path):

        file_name = file_path.split("/")[-1]
        file_name = file_name.split(".")[0]
        output_folder = extract_path + '/' + file_name

        if path.exists(output_folder):
            print('File: {} already extracted.'.format(file_name))
            return output_folder

        print('Extracting file from {} to {}'.format(file_path, extract_path))
        with ZipFile(file_path, 'r') as zipObj:
            # Extract all the contents of zip file in different directory
            zipObj.extractall(extract_path)

        print('File extraction completed.')
        return output_folder

    def create_scheduler_lambda_lr(lambda_fn, optimizer):
        return LambdaLR(optimizer, lr_lambda=[lambda_fn])

    def get_all_file_paths(directory):
        # initializing empty file paths list
        file_paths = []
        filenames = []

        # crawling through directory and subdirectories
        for root, directories, files in os.walk(directory):
            for filename in files:
                # join the two strings in order to form the full filepath.
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)
                filenames.append(filename)

        # returning all file paths
        return file_paths, filenames

    def loadfiles(train_data, test_data):
        final_list = []

        for path in tqdm(train_data):
            final_list.append(Image.open(path))

        for path in tqdm(test_data):
            final_list.append(Image.open(path))

        return np.array(final_list)

    def show(tensors, figsize=(10, 10), *args, **kwargs):
        try:
            tensors = tensors.detach().cpu()
        except:
            pass
        grid_tensor = torchvision.utils.make_grid(tensors, *args, **kwargs)
        # TODO
        grid_image = grid_tensor.permute(1, 2, 0)
        plt.figure(figsize=figsize)
        plt.imshow(grid_image)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def show_pred(tensors, std, mean, *args, **kwargs):
        tensors = (tensors * std[None, :, None, None]) + mean[None, :, None, None]
        from src.utils import Utils
        Utils.show(tensors, *args, **kwargs)

    import subprocess

    proc = subprocess.Popen(["ssh", "-i .ssh/id_rsa", "user@host"],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True,
                            bufsize=0)

    # Fetch output
    for line in proc.stdout:
        print(line.strip())
