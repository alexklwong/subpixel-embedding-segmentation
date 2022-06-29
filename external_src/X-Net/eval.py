import keras
import keras.backend as K
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard
import os
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time
import argparse

from utils import get_score_from_all_slices
from model import create_xception_unet_n
from loss import get_loss, dice
from data import create_train_date_generator, create_val_date_generator, read_data_split

parser = argparse.ArgumentParser()
parser.add_argument('--data_file_path', type=str, default="data/train.h5", help='the h5 file that contains data for training and validation')
parser.add_argument('--pretrained_weight_file', type=str, default='', help='the path to the weight trained by us')
parser.add_argument('--input_shape', type=int, nargs='+', default=(224, 192, 1), help='the input shape to the model')
parser.add_argument('--save_path', type=str, default='', help='the path to save the produced image')
parser.add_argument('--data_order_path', type=str, default='', help='the path to the data_order.txt file')
args = parser.parse_args()

data_file_path = args.data_file_path
pretrained_weights_file = args.pretrained_weight_file
input_shape = args.input_shape
save_path = args.save_path
data_order_path = args.data_order_path

if not os.path.exists(save_path):
    os.mkdir(save_path)

# Create model
K.clear_session()
model = create_xception_unet_n(input_shape=input_shape, pretrained_weights_file=pretrained_weights_file)
# model.compile(optimizer=Adam(lr=1e-3), loss=get_loss, metrics=[dice])


train_patient_indexes, val_patient_indexes = read_data_split("data_split/atlas/traintest", data_order_path=data_order_path)
print(train_patient_indexes)
print(len(train_patient_indexes))
print(val_patient_indexes)
print(len(val_patient_indexes))

with open(data_order_path, 'r') as f:
    data_list = f.readlines()
    data_list = [os.path.split(d.rstrip())[1]+".npy" for d in data_list]


num_slices_val = len(val_patient_indexes) * 189

predicts = []
f = create_val_date_generator(patient_indexes=val_patient_indexes, h5_file_path=data_file_path)
total_time = 0
for patient_index in val_patient_indexes:
    patient_predict = []
    t = time.time()
    for _ in range(189):
        img, label = f.__next__()
        patient_predict.append(model.predict(img))
    total_time += time.time() - t
    patient_predict = np.array(patient_predict)
    patient_save_path = os.path.join(save_path, data_list[patient_index])
    np.save(patient_save_path, patient_predict)
    predicts.append(patient_predict)
predicts = np.array(predicts)
print("total_time_taken: {}".format(total_time))
print("per patient time: {}".format(total_time / 27))

