import keras
import keras.backend as K
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard
import os
import pandas as pd
from sklearn.model_selection import KFold
import time
from tqdm import tqdm
import argparse

from utils import get_score_from_all_slices
from model import create_xception_unet_n
from CLCI_Net import CLCI_Net, dice_coef, dice_coef_loss
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


def train(fold, train_patient_indexes, val_patient_indexes):
    with open(data_order_path, 'r') as f:
        data_list = f.readlines()
        data_list = [os.path.split(d.rstrip())[1]+".npy" for d in data_list]

    num_slices_train = len(train_patient_indexes) * 189
    num_slices_val = len(val_patient_indexes) * 189

    # Create model
    K.clear_session()
    model = CLCI_Net()
    model.load_weights(pretrained_weights_file)
    model.summary()
    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])

    # Evaluate model
    predicts = []
    labels = []
    f = create_val_date_generator(patient_indexes=val_patient_indexes, h5_file_path=data_file_path)
    t = time.time()
    for _ in tqdm(range(num_slices_val)):
        img, label = f.__next__()
        predicts.append(model.predict(img))
        labels.append(label)
    total_time = time.time() - t
    print("total time used: {}".format(total_time))
    print("per patient time: {}".format(total_time / 27))
    predicts = np.array(predicts)
    labels = np.array(labels)
    patient = [data_list[index] for index in val_patient_indexes]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    score_record = get_score_from_all_slices(labels=labels, predicts=predicts, patients=patient, save_path=save_path)

    # print score
    mean_score = {}
    for key in score_record.keys():
        print('In fold ', fold, ', average', key, ' value is: \t ', np.mean(score_record[key]))
        mean_score[key] = np.mean(score_record[key])

    # exit training
    K.clear_session()
    return mean_score


def main():

    train_patient_indexes, val_patient_indexes = read_data_split("data_split/atlas/traintest", data_order_file=data_order_path)
    print(train_patient_indexes)
    print(len(train_patient_indexes))
    print(val_patient_indexes)
    print(len(val_patient_indexes))

    mean_score = train(fold=0, train_patient_indexes=train_patient_indexes, val_patient_indexes=val_patient_indexes)

    # save final score
    # df = pd.DataFrame(mean_score, index=[0])
    # df.to_csv('x_net_final_score.csv', index=False)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()

