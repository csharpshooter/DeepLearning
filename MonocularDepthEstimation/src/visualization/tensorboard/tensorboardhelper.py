from torch.utils.tensorboard import SummaryWriter
import torch

class TensorboardHelper:

    def __init__(self, name):
        self.boardlabel = name
        self.writer = SummaryWriter(self.boardlabel)

    def get_summarywriter(self, name):
        return self.writer

    def add_imagestoplot(self, label, grid):
        self.writer.add_image(label, grid, 0)

    def add_graph(self, model, images):
        self.writer.add_graph(model, images)

    def add_scalars(self, label, datadict, epoch):
        self.writer.add_scalars(label, datadict, epoch)

    def add_scalar(self, label, data, epoch):
        self.writer.add_scalar(label, data, epoch)

    def add_histogram(self, label, data, epoch):
        self.writer.add_histogram(label, data, epoch)

    def set_logdir(self, path):
        self.boardlabel = path

    # def show_board(self):

        # boardcommand =  tensorboard
        # boardcommand = ' '.join('--logdir = ' + self.boardlabel)
        # boardcommand = '\n'.join(boardcommand)
        # print(boardcommand)

    def __del__(self):
        self.writer.close()
