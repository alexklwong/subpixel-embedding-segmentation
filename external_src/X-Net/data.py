'''
data generator
ATLAS dataset has been transformed into .h5 format
'''

import numpy as np
import h5py
from matplotlib import pyplot as plt
import os


def train_data_generator(patient_indexes, h5_file_path, batch_size):
    i = 0
    file = h5py.File(h5_file_path, 'r')
    imgs = file['data']
    labels = file['label']

    # 输入的是病人的index，转换成切片的index
    slice_indexes = []
    for patient_index in patient_indexes:
        for slice_index in range(189):
            slice_indexes.append(patient_index * 189 + slice_index)
    num_of_slices = len(slice_indexes)
    print(num_of_slices)

    while True:
        batch_img = []
        batch_label = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(slice_indexes)

            current_img = imgs[slice_indexes[i]][5:229, 2:194]
            current_label = labels[slice_indexes[i]][5:229, 2:194]
            batch_img.append(current_img)
            batch_label.append(current_label)
            i = (i + 1) % num_of_slices

        yield np.expand_dims(np.array(batch_img), 3), np.expand_dims(np.array(batch_label), 3)


def create_train_date_generator(patient_indexes, h5_file_path, batch_size):
    return train_data_generator(patient_indexes, h5_file_path, batch_size)


def val_data_generator(patient_indexes, h5_file_path, batch_size=1):
    i = 0
    file = h5py.File(h5_file_path, 'r')
    imgs = file['data']
    labels = file['label']

    # 输入的是病人的index，转换成切片的index
    slice_indexes = []
    for patient_index in patient_indexes:
        for slice_index in range(189):
            slice_indexes.append(patient_index * 189 + slice_index)
    num_of_slices = len(slice_indexes)

    while True:
        batch_img = []
        batch_label = []
        for b in range(batch_size):
            current_img = imgs[slice_indexes[i]][5:229, 2:194]
            current_label = labels[slice_indexes[i]][5:229, 2:194]
            batch_img.append(current_img)
            batch_label.append(current_label)
            i = (i + 1) % num_of_slices
        yield np.expand_dims(np.array(batch_img), 3), np.expand_dims(np.array(batch_label), 3)


def create_val_date_generator(patient_indexes, h5_file_path, batch_size=1):
    return val_data_generator(patient_indexes, h5_file_path, batch_size)


def read_data_split(data_split_path, data_order_path):
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


def get_data_to_index_dict(file):
    data = []
    with open(file, 'r') as f:
        data = f.readlines()

    data_to_index_dict = {d.strip():i for i, d in enumerate(data)}
    return data_to_index_dict
