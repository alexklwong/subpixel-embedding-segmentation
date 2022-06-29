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

from utils import get_score_from_all_slices
from model import create_xception_unet_n
from CLCI_Net import CLCI_Net, dice_coef, dice_coef_loss
from loss import get_loss, dice
from data import create_train_date_generator, create_val_date_generator, read_data_split

data_file_path = 'data/train.h5'
pretrained_weights_file = None
input_shape = (224, 192, 1)
batch_size = 8
num_folds = 5


def train(fold, train_patient_indexes, val_patient_indexes):
    with open('data/data_order.txt', 'r') as f:
        data_list = f.readlines()
        data_list = [os.path.split(d.rstrip())[1]+".npy" for d in data_list]

    log_dir = "trained_model/trial2/"
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    num_slices_train = len(train_patient_indexes) * 189
    num_slices_val = len(val_patient_indexes) * 189

    # Create model
    K.clear_session()
    model = CLCI_Net()
    model.load_weights('trained_model/trial2/trial2ep=025-loss=0.303-val_loss=0.198.h5')
    model.summary()
    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])

    # Get callbacks
    checkpoint = ModelCheckpoint(log_dir + 'ep={epoch:03d}-loss={loss:.3f}-val_loss={val_loss:.3f}.h5', verbose=1,
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_delta=1e-3, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)
    csv_logger = CSVLogger(log_dir + 'record.csv')
    tensorboard = TensorBoard(log_dir=log_dir)

    # train the model
    # model.fit_generator(
    #     create_train_date_generator(patient_indexes=train_patient_indexes, h5_file_path=data_file_path, batch_size=batch_size),
    #     steps_per_epoch=max(1, num_slices_train // batch_size),
    #     validation_data=create_val_date_generator(patient_indexes=val_patient_indexes, h5_file_path=data_file_path, batch_size=9),
    #     validation_steps=max(1, num_slices_val // 9),
    #     epochs=100,
    #     initial_epoch=0,
    #     callbacks=[checkpoint, reduce_lr, early_stopping, tensorboard, csv_logger])
    # model.save_weights(log_dir + 'trained_final_weights.h5')

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
    score_record = get_score_from_all_slices(labels=labels, predicts=predicts, patients=patient)

    # save score
    # df = pd.DataFrame(score_record)
    # df.to_csv(os.path.join(log_dir, 'score_record.csv'), index=False)

    # print score
    mean_score = {}
    for key in score_record.keys():
        print('In fold ', fold, ', average', key, ' value is: \t ', np.mean(score_record[key]))
        mean_score[key] = np.mean(score_record[key])

    # exit training
    K.clear_session()
    return mean_score


def main():

    # prepare indexes of patients for training and validation, respectively
    # num_patients = 229
    # patients_indexes = np.array([i for i in range(num_patients)])
    # kf = KFold(n_splits=num_folds, shuffle=False)


    # train, and record the scores of each fold
    # folds_score = []
    # for fold, (train_patient_indexes, val_patient_indexes) in enumerate(kf.split(patients_indexes)):
    #     fold_mean_score = train(fold=fold, train_patient_indexes=train_patient_indexes, val_patient_indexes=val_patient_indexes)
    #     folds_score.append(fold_mean_score)
    train_patient_indexes, val_patient_indexes = read_data_split("data_split/atlas/train_val")
    print(train_patient_indexes)
    print(len(train_patient_indexes))
    print(val_patient_indexes)
    print(len(val_patient_indexes))

    mean_score = train(fold=0, train_patient_indexes=train_patient_indexes, val_patient_indexes=val_patient_indexes)

    # calculate average score
    # print('Final score from ', num_folds, ' folds cross validation:')
    # final_score = {}
    # for key in folds_score[0].keys():
    #     scores = []
    #     for i in range(num_folds):
    #         scores.append(folds_score[i][key])
    #     final_score[key] = np.mean(scores)
    #     print(key, ' score: \t', final_score[key])


    # save final score
    df = pd.DataFrame(mean_score, index=[0])
    df.to_csv('x_net_final_score.csv', index=False)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()

