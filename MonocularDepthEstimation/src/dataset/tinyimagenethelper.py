# from src.utils import Utils
import time
import numpy as np
from tqdm import tqdm
from src.preprocessing import PreprocHelper


class TinyImagenetHelper:

    def __init__(self):
        self.output_folder = ''
        self.dictionaryFileName = ''

    def download_dataset(self, folder_path):
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        from src.utils import Utils
        folder_path = Utils.download_file(folder_path, url=url)

        return Utils.extract_zip_file(file_path=folder_path, extract_path='data')

    def get_id_dictionary(self, path):
        id_dict = {}
        for i, line in enumerate(open(path + '/wnids.txt', 'r')):
            id_dict[line.replace('\n', '')] = i
        return id_dict

    def get_class_to_id_dict(self, id_dict, path):
        # id_dict = self.get_id_dictionary(path)
        all_classes = {}
        class_labels = []
        result = {}
        for i, line in enumerate(open(path + '/words.txt', 'r')):
            n_id, word = line.split('\t')[:2]
            all_classes[n_id] = word
            class_labels.append(word)
        for key, value in id_dict.items():
            result[value] = (key, all_classes[key])
        return result, class_labels

    def get_train_test_labels_data(self, id_dict, path, test_split=0.3):
        print('Starting data loading')

        train_data, test_data = [], []
        train_labels, test_labels = [], []
        t = time.time()
        total_val_images = 10000
        images_for_class = 500
        train_image_count = images_for_class - (images_for_class * test_split)
        for key, value in tqdm(id_dict.items()):
            all_data, all_labels = [], []
            for i in range(images_for_class):
                all_data.append(path + '/train/{}/images/{}_{}.JPEG'.format(key, key, str(i)))
                all_labels.append(id_dict[key])

            for x in range(0, images_for_class):
                if x < train_image_count:
                    train_data.append(all_data[x])
                    train_labels.append(all_labels[x])
                else:
                    test_data.append(all_data[x])
                    test_labels.append(all_labels[x])

        test_image_count = total_val_images - (total_val_images * test_split)
        val_count = 0
        for line in tqdm(open(path + '/val/val_annotations.txt')):
            img_name, class_id = line.split('\t')[:2]
            if val_count < test_image_count:
                train_data.append(path + '/val/images/{}'.format(img_name))
                train_labels.append(id_dict[class_id])
            else:
                test_data.append(path + '/val/images/{}'.format(img_name))
                test_labels.append(id_dict[class_id])

            val_count += 1

        print('Finished data loading, in {} seconds'.format(time.time() - t))
        return np.array(train_data), train_labels, np.array(test_data), test_labels

    def get_classes(self, path):
        id_dict = self.get_id_dictionary(path=path)
        values, class_labels = self.get_class_to_id_dict(id_dict=id_dict, path=path)
        return values, id_dict

    def get_tiny_image_net_test_train_loader(self, id_dict, path, batch_size):
        train_data, train_label, test_data, test_label = self.get_train_test_labels_data(id_dict, path)

        # from src.utils import tiny_image_net_mean
        # from src.utils import tiny_image_net_std
        # compose_train, compose_test = PreprocHelper.getalbumentationstraintesttransforms(
        #     tiny_image_net_mean, tiny_image_net_std)

        # from src.utils import tiny_image_net_mean
        # from src.utils import tiny_image_net_std
        # compose_train, compose_test = PreprocHelper.getpytorchtransforms(
        #     tiny_image_net_mean, tiny_image_net_std)

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        from src.utils import image_net_mean
        from src.utils import image_net_std
        compose_train, compose_test = PreprocHelper.getpytorchtransforms(
            mean, std)

        from src.dataset import Dataset
        ds = Dataset()
        train_dataset = ds.get_tiny_imagenet_train_dataset(train_image_data=train_data, train_image_labels=train_label,
                                                           train_transforms=compose_train)
        test_dataset = ds.get_tiny_imagenet_test_dataset(test_image_labels=test_label, test_image_data=test_data,
                                                         test_transforms=compose_test)

        from src.dataset import Dataloader
        data_loader = Dataloader(traindataset=train_dataset, testdataset=test_dataset, batch_size=batch_size)
        train_loader = data_loader.gettraindataloader()
        test_loader = data_loader.gettestdataloader()

        return test_loader, train_loader
