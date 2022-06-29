from codecs import encode
import h5py
import os
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--size', nargs='+', type=int, default=(128,128), help='the size of the image to reshape to')
parser.add_argument('--data_path', type=str, default='external_src/KiU-Net/data/train.h5', help='path to h5 file that contains all data')
parser.add_argument('--train_path', type=str, default='external_src/KiU-Net/data/train',help='path to store the train data')
parser.add_argument('--validation_path', type=str, default='external_src/KiU-Net/data/validation', help='path to store the validation data')
parser.add_argument('--data_order_path', type=str, default='external_src/KiU-Net/data/data_order.txt', help='path to text file that contains the order of data in the h5 file')

args = parser.parse_args()

data_path = args.data_path
train_path = args.train_path
validation_path = args.validation_path
data_order_path = args.data_order_path
size = args.size

def get_data_to_index_dict(file):
    data = []
    with open(file, 'r') as f:
        data = f.readlines()

    data_to_index_dict = {d.strip():i for i, d in enumerate(data)}
    return data_to_index_dict

def read_data_split(data_split_path):
    train_split_path = os.path.join(data_split_path, 'training')
    train_image_paths = os.path.join(train_split_path, "train_scans.txt")
    val_split_path = os.path.join(data_split_path, "testing")
    val_image_paths = os.path.join(val_split_path, "test_scans.txt")

    data_to_index_dict = get_data_to_index_dict(data_order_path)
    print(data_to_index_dict)

    train_data = []
    with open(train_image_paths, "r") as train_data_file:
        train_data = train_data_file.readlines()

    train_data = [os.path.split(data)[0] for data in train_data]
    train_data = [os.path.join("data/atlas/atlas_standard", data) for data in train_data]

    train_index = [data_to_index_dict[i] for i in train_data]

    val_data = []
    with open(val_image_paths, "r") as val_data_file:
        val_data = val_data_file.readlines()

    val_data = [os.path.split(data)[0] for data in val_data]
    val_data = [os.path.join("data/atlas/atlas_standard", data) for data in val_data]

    val_index = [data_to_index_dict[i] for i in val_data]

    return train_index, val_index

if __name__ == "__main__":
    data_file = h5py.File(data_path, 'r')
    data = data_file['data']
    label = data_file['label']

    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(validation_path):
        os.mkdir(validation_path)

    with open(data_order_path, 'r') as f:
        data_list = f.readlines()
        data_list = [os.path.split(d.rstrip())[1]+".npy" for d in data_list]


    train_index, val_index = read_data_split('data_split/atlas/traintest')
    train_count = 0

    train_img_path = os.path.join(train_path, 'img')
    train_label_path = os.path.join(train_path, 'label')

    val_img_path = os.path.join(validation_path, 'img')
    val_label_path = os.path.join(validation_path, 'label')

    paths = [train_img_path, train_label_path, val_img_path, val_label_path]
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

    for index in train_index:
        patient_image = data[index*189:(index + 1)*189]
        patient_label = label[index*189:(index + 1)*189]
        patient_name = data_list[index]

        for i, (img, lb) in enumerate(zip(patient_image, patient_label)):
            train_count += 1
            print('processing {}/{}'.format(train_count, len(train_index) * 189), end='\r')

            img = ((img - img.min()) / img.max() * 255).astype(np.uint8)
            img[np.isnan(img)] = 0
            assert np.any(np.isnan(img)) == False
            lb[lb >= 0.5] = 1
            lb[lb < 0.5] = 0
            lb = lb.astype(np.uint8)

            img = Image.fromarray(img)
            lb = Image.fromarray(lb)

            img = img.resize(size, resample=Image.NEAREST)
            lb = lb.resize(size, resample=Image.NEAREST)

            train_img_file_path = '{}.png'.format(os.path.join(train_img_path, '{:05d}'.format(train_count)))
            train_label_file_path = '{}.png'.format(os.path.join(train_label_path, '{:05d}'.format(train_count)))

            img.save(train_img_file_path)
            lb.save(train_label_file_path)

    for j, index in enumerate(val_index):
        patient_image = data[index*189:(index + 1)*189]
        patient_label = label[index*189:(index + 1)*189]
        patient_name = data_list[index]
        print('processing patient: {}'.format(patient_name[:-4]))

        for i, (img, lb) in enumerate(zip(patient_image, patient_label)):
            img = ((img - img.min()) / img.max() * 255).astype(np.uint8)
            img[np.isnan(img)] = 0
            assert np.any(np.isnan(img)) == False
            lb[lb >= 0.5] = 1
            lb[lb < 0.5] = 0
            lb = lb.astype(np.uint8)


            img = Image.fromarray(img)
            lb = Image.fromarray(lb)

            img = img.resize(size, resample=Image.NEAREST)
            lb = lb.resize(size, resample=Image.NEAREST)

            val_img_file_path = '{}.png'.format(os.path.join(val_img_path, '{}_{:03d}'.format(patient_name[:-4], i)))
            val_label_file_path = '{}.png'.format(os.path.join(val_label_path, '{}_{:03d}'.format(patient_name[:-4], i)))

            img.save(val_img_file_path)
            lb.save(val_label_file_path)