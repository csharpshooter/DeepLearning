import cv2
from tqdm import tqdm
import os
from src.utils.utils import Utils
from PIL.Image import Image


class MonocularHelper:

    def __init__(self):
        self.output_folder = ''
        self.dictionaryFileName = ''

    def get_train_test_data(self, masks_folder, depth_masks_folder, images_folder, no_of_batches, total_images_count,
                            bg_folder, train_split=0.7):
        batch = 1
        batch_folder_name = r'batch_'

        bg_ext = '.jpg'

        train_imgs = []
        test_imgs = []

        train_labels = []
        test_labels = []

        train_count = total_images_count * train_split

        master_count = 0
        for x in tqdm(range(0, no_of_batches)):

            if not x == 0:
                batch += 1

            current_img_path = images_folder + os.path.sep + batch_folder_name + str(batch)
            current_masks_path = masks_folder + os.path.sep + batch_folder_name + str(batch)
            current_depth_masks_path = depth_masks_folder + os.path.sep + batch_folder_name + str(batch)

            path_imgs, name_imgs = Utils.get_all_file_paths(current_img_path)
            path_masks, name_masks = Utils.get_all_file_paths(current_masks_path)
            path_dms, name_dms = Utils.get_all_file_paths(current_depth_masks_path)

            for count in (range(0, len(path_imgs))):

                img_path = path_imgs[count]

                bg_name = name_imgs[count].split('_')[0]

                bg_path = bg_folder + os.path.sep + bg_name + bg_ext

                target = {"labels": name_imgs[count], "masks": path_masks[count], "image_id": count, "bg_path": bg_path,
                          "depth_mask": path_dms[count]}

                if master_count < train_count:
                    train_imgs.append(img_path)
                    train_labels.append(target)
                else:
                    test_imgs.append(img_path)
                    test_labels.append(target)

                master_count += 1

        return train_imgs, train_labels, test_imgs, test_labels

    def resize(self, img):
        scale_percent = 29  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized
