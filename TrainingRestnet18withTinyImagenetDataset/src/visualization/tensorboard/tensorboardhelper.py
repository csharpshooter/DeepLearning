from torch.utils.tensorboard import SummaryWriter


class TensorboardHelper:

    def __init__(self, name):
        self.boardlabel = name
        self.writer = SummaryWriter(self.boardlabel)

    def getsummarywriter(self, name):
        return self.writer

    def addimagestoplot(self, label, grid):
        self.writer.add_image(label, grid, 0)

    def addgraph(self, model, images):
        self.writer.add_graph(model, images)

    def addscalars(self, label, datadict, epoch):
        self.writer.add_scalars(label, datadict, epoch)

    def addscalar(self, label, data, epoch):
        self.writer.add_scalar(label, data, epoch)

    def addhistogram(self, label, data, epoch):
        self.writer.add_histogram(label, data, epoch)

    def setlogdir(self, path):
        self.boardlabel = path

    # def showboard(self):
    #     boardcommand =  tensorboard
    #     boardcommand = ' '.join('--logdir = ' + self.boardlabel)
    #     boardcommand = '\n'.join(boardcommand)
    #     print(boardcommand)

    def __del__(self):
        self.writer.close()
