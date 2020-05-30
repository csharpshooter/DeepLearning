import torch


class Dataloader(object):

    def __init__(self, traindataset, testdataset, batch_size=64):
        self.traindataset = traindataset
        self.testdataset = testdataset
        self.batch_size = batch_size
        self.num_workers = 4
        self.pin_memory = True
        self.shuffle = True

        print(self.batch_size)

    def gettraindataloader(self):
        return torch.utils.data.DataLoader(dataset=self.traindataset, batch_size=self.batch_size,
                                           num_workers=self.num_workers, shuffle=self.shuffle,
                                           pin_memory=self.pin_memory)

    def gettestdataloader(self):
        return torch.utils.data.DataLoader(dataset=self.testdataset, batch_size=self.batch_size,
                                           num_workers=self.num_workers, shuffle=self.shuffle,
                                           pin_memory=self.pin_memory)
