# TODO: rename file to make it more descriptive
import os

import numpy as np
from matplotlib.pyplot import imread
from skimage.transform import resize


def read_imgs_per_website(data_path, targets, imgs_num, reshape_size, start_target_count):
    all_imgs = np.zeros(shape=[imgs_num, 224, 224, 3])
    all_labels = np.zeros(shape=[imgs_num, 1])

    all_file_names = []
    targets_list = targets.splitlines()
    count = 0
    for i in range(0, len(targets_list)):
        target_path = data_path / targets_list[i]
        print(target_path)
        file_names = sorted(os.listdir(target_path))
        for j in range(0, len(file_names)):
            try:
                img = imread(target_path / file_names[j])
                img = img[:, :, 0:3]
                all_imgs[count, :, :, :] = resize(img, (reshape_size[0], reshape_size[1]), anti_aliasing=True)
                all_labels[count, :] = i + start_target_count
                all_file_names.append(file_names[j])
                count = count + 1
            except:  # noqa: E722
                # some images were saved with a wrong extensions
                try:
                    img = imread(target_path / file_names[j], format="jpeg")
                    img = img[:, :, 0:3]
                    all_imgs[count, :, :, :] = resize(img, (reshape_size[0], reshape_size[1]), anti_aliasing=True)
                    all_labels[count, :] = i + start_target_count
                    all_file_names.append(file_names[j])
                    count = count + 1
                except:  # noqa: E722
                    print("failed at:")
                    print("***")
                    print(file_names[j])
                    break
    return all_imgs, all_labels, all_file_names
