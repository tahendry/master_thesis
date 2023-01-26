from keras.layers import Input, Conv3D, MaxPooling3D, Dropout, BatchNormalization, Reshape, Dense, ELU, concatenate, add, Lambda, MaxPooling3D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
from keras.metrics import binary_crossentropy
import numpy as np

from sepconv3D import SeparableConv3D   #import from https://github.com/simeon-spasov/MCI/tree/master/utils/sepConv3D.py

from numpy.random import seed
seed(2018)
import random
random.seed(2018)
from tensorflow import set_random_seed
set_random_seed((2018))
import os
import sys
import gc
import numpy as np
import scipy.io as sio
import sklearn.metrics
import keras
from collections import Counter
import time
from keras.layers import Dense, Input
from keras.models import Model, Sequential
from keras import regularizers
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, LSTM, Conv3D, MaxPool3D, Conv1D, MaxPool1D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras import optimizers
from keras import losses
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA, IncrementalPCA
from keras import backend as K
import tensorflow as tf

raw_path = r'/data/kuang/hcp/tfMRI_motor/'
procData_path = os.path.join(raw_path, 'proc_hcp/stdProc_meanPSC')   
subname_file = os.path.join(raw_path, 'subject_name995.txt')   
subject_list = np.genfromtxt(subname_file, dtype=str).tolist()
condition_name = ['lf', 'lh', 'rf', 'rh', 't']           
movement_rep = 2             
mat_file_name = 'z_sub_%s_motor'

drop_rate = 0.5
w_regularizer = None #regularizers.l2(5e-5)

DIMX = 91
DIMY = 109
DIMZ = 91
numClass = 5
num_use_subject = 995       #  200, 500, 995
num_mem_subject = 500       #  160, 400, 500
samplePerSubject = int(len(condition_name) * movement_rep)

def main():
    k = 5
    use_subname_file = subject_list[:num_use_subject]        
    train_set, test_set, valid_set = randomSplit_TrainAndTest(use_subname_file, k)

    all_result_value = []

    #cross validation

    for i in range(k):
        print('##### Round ', i, ' -------------')
        train_name = train_set[i]
        test_name = test_set[i]
        valid_name = valid_set[i]

        result = sep3d_method(train_name, test_name, valid_name, i)
        gc.collect()

        all_result_value.append(result[:-1])
    

    print('each : test_acc, num_hit, test_precision, test_recall, test_f1, durTime')
    print('sepconv 3D : ', all_result_value)
    print('sepconv 3D mean: ', np.mean(all_result_value, axis=0))
    print('sepconv 3D std: ', np.std(all_result_value, axis=0))

    print('Finished.')

def sep3d_method(train_name, test_name, valid_name, round_index):
    print('*** Run sep_CNN3D ...')

    xalex3D = XAlex3D(w_regularizer=w_regularizer, drop_rate=drop_rate)
    input = Input(shape=(DIMX, DIMY, DIMZ, 1))
    fc = xalex3D (input)
    output = Dense(units=5, activation='softmax', name='pred')(fc)
    model = Model(inputs=input, outputs=output)

    print(model.summary())

    my_adam = optimizers.Adam(lr=0.0025, beta_1=0.9, beta_2=0.999, amsgrad=True)
    model.compile(loss='categorical_crossentropy', optimizer=my_adam, metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='loss', patience=6, mode='auto')
    # checkpoint
    bestModelSavePath = '/data/kuang/hcp_clj/para/sep3d_%s_weights_best_500.hdf5' % str(round_index)
    checkpoint = ModelCheckpoint(bestModelSavePath, monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='auto')
    lrate = LearningRateScheduler(sch_func, verbose=1)

    # load some train data to memory
    mem_train_name = train_name[:num_mem_subject]
    disk_train_name = set(train_name[num_mem_subject:])
    mem_train_x, mem_train_y = [], []
    for name in mem_train_name:
        datafile_name = os.path.join(procData_path, mat_file_name % name)
        datafile = sio.loadmat(datafile_name)
        for (j, cd) in enumerate(condition_name):
            for i_rep in range(movement_rep):
                sheetname = cd + '_' + str(i_rep)
                mem_train_x.append(datafile[sheetname])
                mem_train_y.append(j)
        del datafile
    gc.collect()
    # load valid data to memory
    valid_x, valid_y = [], []
    for name in valid_name:
        datafile_name = os.path.join(procData_path, mat_file_name % name)
        datafile = sio.loadmat(datafile_name)
        for (j, cd) in enumerate(condition_name):
            for i_rep in range(movement_rep):
                sheetname = cd + '_' + str(i_rep)
                valid_x.append(datafile[sheetname])
                valid_y.append(j)
        del datafile
    gc.collect()

    # generator
    def generator_CNN3D_MandD(subject_name0, batch_size, train_valid_test_flag):
        # train_valid_test_flag: Train, Valid, Test, Info
        subject_name = np.array(subject_name0)
        if train_valid_test_flag in ['Train']:
            while True:
                np.random.shuffle(subject_name)
                mem_random_index = [xx for xx in range(samplePerSubject * num_mem_subject)]
                np.random.shuffle(mem_random_index)
                mem_visit_time, disk_visit_time = 0, 0
                batch_subject_size = int(batch_size / samplePerSubject)
                itr_time = int(len(subject_name) / batch_subject_size)
                for itr in range(itr_time):
                    subjectNameBatch = subject_name[itr * batch_subject_size: (itr + 1) * batch_subject_size]
                    data_x = []
                    data_y = []
                    for name in subjectNameBatch:
                        if name in disk_train_name:
                            datafile_name = os.path.join(procData_path, mat_file_name % name)
                            datafile = sio.loadmat(datafile_name)
                            for (j, cd) in enumerate(condition_name):
                                for i_rep in range(movement_rep):
                                    sheetname = cd + '_' + str(i_rep)
                                    data_x.append(datafile[sheetname])
                                    data_y.append(j)
                            del datafile
                            disk_visit_time += 1
                        else:
                            mem_itr_index = mem_random_index[
                                            mem_visit_time * samplePerSubject: (mem_visit_time + 1) * samplePerSubject]
                            for m_index in mem_itr_index:
                                data_x.append(mem_train_x[m_index])
                                data_y.append(mem_train_y[m_index])
                            mem_visit_time += 1
                        gc.collect()
                    shuffle_tmp = list(zip(data_x, data_y))
                    np.random.shuffle(shuffle_tmp)
                    data_x, data_y = zip(*shuffle_tmp)
                    del shuffle_tmp
                    gc.collect()
                    # data transform rwt method
                    data_x = np.array(data_x)
                    dnum, dx, dy, dz = data_x.shape
                    data_x = data_x.reshape(dnum, dx, dy, dz, 1)
                    data_y = to_categorical(np.array(data_y), num_classes=numClass)
                    yield (data_x, data_y)
                    del data_x, data_y
                    gc.collect()
                gc.collect()
        elif train_valid_test_flag in ['Valid']:
            while True:
                mem_random_index = [xx for xx in range(len(valid_y))]
                np.random.shuffle(mem_random_index)
                itr_time = int(len(valid_y) / batch_size)
                for itr in range(itr_time):
                    data_x, data_y = [], []
                    mem_itr_index = mem_random_index[itr * batch_size: (itr + 1) * batch_size]
                    for m_index in mem_itr_index:
                        data_x.append(valid_x[m_index])
                        data_y.append(valid_y[m_index])
                    # data transform rwt method
                    data_x = np.array(data_x)
                    dnum, dx, dy, dz = data_x.shape
                    data_x = data_x.reshape(dnum, dx, dy, dz, 1)
                    data_y = to_categorical(np.array(data_y), num_classes=numClass)
                    yield (data_x, data_y)
                    del data_x, data_y
                    gc.collect()
                gc.collect()
        elif train_valid_test_flag in ['Test']:
            test_num = len(subject_name)
            data_x = []
            data_y = []
            for ti in range(test_num):
                subname = subject_name[ti]
                datafile_name = os.path.join(procData_path, mat_file_name % subname)
                datafile = sio.loadmat(datafile_name)
                for (j, cd) in enumerate(condition_name):
                    for i_rep in range(movement_rep):
                        sheetname = cd + '_' + str(i_rep)
                        data_x.append(datafile[sheetname])
                        data_y.append(j)
                del datafile
                if (len(data_y) == batch_size) or (ti == (test_num - 1)):
                    # data transform rwt method
                    data_x = np.array(data_x)
                    dnum, dx, dy, dz = data_x.shape
                    data_x = data_x.reshape(dnum, dx, dy, dz, 1)
                    data_y = to_categorical(np.array(data_y), num_classes=numClass)
                    yield (data_x, data_y)
                    del data_x, data_y
                    data_x, data_y = [], []
                gc.collect()
            print('Test generator finished. ')
            del data_x, data_y
            gc.collect()
        else:
            subname = subject_name[0]
            datafile_name = os.path.join(procData_path, mat_file_name % subname)
            datafile = sio.loadmat(datafile_name)
            sheetname = condition_name[0] + '_0'
            one_data = datafile[sheetname]
            dx, dy, dz = one_data.shape
            one_data = one_data.reshape(dx, dy, dz, 1)
            print('Show data info: ', one_data.shape)
            del datafile, one_data
            gc.collect()
            return

    generator_CNN3D_MandD(train_name, 0, 'Info')

    batch_size = 20
    stepTrain = int(len(train_name) * samplePerSubject / batch_size)
    stepValid = int(len(valid_name) * samplePerSubject / batch_size)
    print('^^^^^^^^^^^^^^^^', stepTrain, stepValid)

    time_callback = TimeHistory()
    startTime = time.time()
    model.fit_generator(generator=generator_CNN3D_MandD(train_name, batch_size, 'Train'),
                        steps_per_epoch=stepTrain,
                        epochs=60, verbose=1, callbacks=[early_stop, checkpoint, lrate, time_callback],
                        validation_data=generator_CNN3D_MandD(valid_name, batch_size, 'Valid'),
                        validation_steps=stepValid)
    endTime = time.time()
    durTime = endTime - startTime
    print('&&& sepconv3D Time: ', durTime)
    print("time_callback.times:", time_callback.times)
    print("time_callback.totaltime:", time_callback.totaltime)
    del mem_train_x, mem_train_y, valid_x, valid_y
    gc.collect()

    model.load_weights(bestModelSavePath)

    # test
    all_pred, all_true_y = None, None
    predNoneFlag = True
    print('Start Predict.')
    for test_item in generator_CNN3D_MandD(test_name, batch_size, 'Test'):
        test_x, test_y = test_item[0], test_item[1]
        pred = model.predict(test_x)
        if predNoneFlag:
            all_pred = pred
            all_true_y = test_y
            predNoneFlag = False
        else:
            all_pred = np.concatenate((all_pred, pred), axis=0)
            all_true_y = np.concatenate((all_true_y, test_y), axis=0)
    print(all_pred.shape, all_true_y.shape)

    test_y_index, pred_index = [], []
    hit = 0
    for j in range(len(all_true_y)):
        ty_ind = int(np.where(all_true_y[j] == np.max(all_true_y[j]))[0])
        pd_ind = int(np.where(all_pred[j] == np.max(all_pred[j]))[0])
        print(ty_ind, '   ', pd_ind, '   ', all_true_y[j], all_pred[j])
        if ty_ind == pd_ind:
            hit += 1
        test_y_index.append(ty_ind)
        pred_index.append(pd_ind)
    acc_test = hit / len(all_true_y)
    print(acc_test)
    print(model.summary())

    test_acc, test_precision, test_recall, test_f1, test_cm = vote2calAcc(pred_index, test_y_index)

    K.clear_session()
    # clear memory
    for xvar in locals().keys():
        if xvar not in ['test_acc', 'hit', 'test_precision', 'test_recall', 'test_f1', 'durTime', 'test_cm']:
            del locals()[xvar]
    gc.collect()
    return test_acc, hit, test_precision, test_recall, test_f1, durTime, test_cm

# refering to the code https://github.com/simeon-spasov/MCI
def XAlex3D(w_regularizer=None, drop_rate=0.):
    def f(fmri_volume):
        conv_mid_1 = mid_flow(fmri_volume, w_regularizer, filters=16)

        # Flatten 3D conv network representations
        flat = Flatten()(conv_mid_1)

        fc = _fc_bn_relu_drop(128, w_regularizer, drop_rate=drop_rate)(flat)
        return fc
    return f

def _fc_bn_relu_drop(units, w_regularizer=None, drop_rate=0., name=None):
    # Defines Fully connected block 
    def f(input):
        fc = Dense(units=units, kernel_initializer=initializers.glorot_normal(),kernel_regularizer=w_regularizer, name=name)(input)
        fc = LeakyReLU()(fc)
        fc = BatchNormalization()(fc)
        fc = Dropout(drop_rate)(fc)
        return fc

    return f

def _conv_bn_relu_pool_drop(filters, height, width, depth, strides=(1, 1, 1), padding='same', w_regularizer=None,
                            drop_rate=None, name=None, pool=False):
    # Defines convolutional block
    def f(input):
        conv = Conv3D(filters, (height, width, depth),
                        strides=strides, kernel_initializer=initializers.glorot_normal(),
                        padding=padding, kernel_regularizer=w_regularizer, name=name)(input)
        elu = LeakyReLU()(conv)
        if pool == True:
            elu = MaxPooling3D(pool_size=2, strides=2, padding=padding)(elu)
        norm = BatchNormalization()(elu)
        return norm

    return f

def _sepconv_bn_relu_pool_drop(filters, height, width, depth, strides=(1, 1, 1), padding='same', depth_multiplier=1,
                                w_regularizer=None, name=None, pool=False):
    # Defines separable convolutional block
    def f(input):
        sep_conv = SeparableConv3D(filters, (height, width, depth),
                                    strides=strides, depth_multiplier=depth_multiplier,
                                    kernel_initializer=initializers.glorot_normal(),
                                    padding=padding, kernel_regularizer=w_regularizer, name=name)(input)
        elu = LeakyReLU()(sep_conv)
        if pool == True:
            elu = MaxPooling3D(pool_size=2, strides=2, padding=padding)(elu)
        norm = BatchNormalization()(elu)
        return norm

    return f

def mid_flow(x, w_regularizer, filters=96):
    # 3 consecutive separable blocks with a residual connection (refer to fig. 4)
    x = _conv_bn_relu_pool_drop(filters, 3, 3, 3, padding='valid',
                                    w_regularizer=w_regularizer, pool=True)(x)
    x = _sepconv_bn_relu_pool_drop(32, 3, 3, 3, padding='valid',  depth_multiplier=1,
                                    w_regularizer=w_regularizer, pool=True)(x)

    return x



def sch_func(epoch, lr):
    init_lr = 0.0025
    drop = 0.5
    epochs_drop = 50
    new_lr = init_lr * (drop ** np.floor(epoch / epochs_drop))
    return new_lr


# split train and test set
def randomSplit_TrainAndTest(sl, k):
    np.random.seed(2018)
    np.random.shuffle(sl)
    print(sl)
    part_num = int(len(sl) / k)
    train_set = []
    test_set = []
    valid_set = []
    for i in range(k):
        test_part = []
        if i == k - 1:
            test_part = sl[i*part_num :]
        else:
            test_part = sl[i*part_num : (i+1)*part_num]
        test_valid_splitValue = int(len(test_part)/2)
        test_set.append(test_part[test_valid_splitValue:])
        valid_set.append(test_part[:test_valid_splitValue])
        train_set.append(list(set(sl) - set(test_part)))
    return train_set, test_set, valid_set

# evaluation method
def vote2calAcc(pred_cl, test_y_cl):
    pred_cl = np.array(pred_cl)
    test_y_cl = np.array(test_y_cl)
    n_test = len(test_y_cl)
    n_hit = 0
    for test_i in range(n_test):
        if test_y_cl[test_i] == pred_cl[test_i]:
            n_hit += 1
    test_acc = n_hit / n_test
    test_precision = sklearn.metrics.precision_score(test_y_cl, pred_cl, average='weighted')
    test_recall = sklearn.metrics.recall_score(test_y_cl, pred_cl, average='weighted')
    test_f1 = sklearn.metrics.f1_score(test_y_cl, pred_cl, average='weighted')
    test_cm = sklearn.metrics.confusion_matrix(test_y_cl, pred_cl).T
    print('***Log-result- ', test_acc, test_precision, test_recall, test_f1)
    print(test_cm)
    return test_acc, test_precision, test_recall, test_f1, test_cm

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.totaltime = time.time()

    def on_train_end(self, logs={}):
        self.totaltime = time.time() - self.totaltime

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


if __name__ == '__main__':
    main()



