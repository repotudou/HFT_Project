import numpy as np
import tensorflow as tf
import time
import copy
import os
import sys
import h5py
import random
import pandas as pd
from MultiCNN_model import MultiCNN_Model
from MultiLSTM_model import MultiLSTM_Model
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import math_ops
from tensorflow.contrib import layers as layers_lib

time1 = time.time()
LR = .001
epsilonADAM = 1e-8
BPTT_length=80
num_nodes = 1
stock_num = 0
num_levels = 10
num_inputs = num_levels*2+4
num_stocks = 1
batches = 500
num_classes=3
cnn_filter_size=3
pooling_filter_size=2
num_filters_per_size=(64,128,256,512)
# num_rep_block=(2,2,2,2) # how many layers in each block??
num_rep_block=(1,1,1,1) 
epoch_limit = 1
keep_prob_train_1=0.95
keep_prob_train_01=0.95
T = 1000*batches
T_eval = 100*batches
levels = 10
time_interval_ratio = 10


folder = '/home/leifan/Data/1Y/20Stocks_LoadRNNdata_1/'
stock_name = 'AMD'
stock_file_name = stock_name+'_loadRNN_1.hdf5'
model_identifier = stock_name

HDF5_file = h5py.File(folder + stock_file_name, 'r')
X_np_eval_1 = HDF5_file['X_test']
Y_np_eval_1 = HDF5_file['y_test']

folder_01 = '/home/leifan/Data/1Y/20Stocks_LoadRNNdata_01/'
# folder = '/Users/leifan/Dropbox/VDCNN/'
stock_file_name_01 = stock_name+'_loadRNN_01.hdf5'
HDF5_file_01 = h5py.File(folder_01 + stock_file_name_01, 'r')
X_np_eval_01 = HDF5_file_01['X_test']
Y_np_eval_01 = HDF5_file_01['y_test']




store_path = '/home/leifan/Result/Comb_CNN_LSTM/'


cond_acc = np.zeros(epoch_limit)

for i in range(epoch_limit):
    print('epoch: %d' %(i+1))

    # EVALUATION ERROR
    current_eval_error = 0.0
    current_eval_acc = 0.0
    random_index_list = []
    total_movements=0.0
    for k in range(batches):
        random_index = random.randint(0,len(Y_np_eval)-T_eval-1)
        random_index_list.append(random_index)
    counter = 0
    x_train_batch_1 = np.float32(np.zeros((batches, num_inputs,BPTT_length)))
    y_train_batch_1 = np.int64(np.zeros((batches)))
    x_train_batch_01 = np.float32(np.zeros((batches, num_inputs,BPTT_length)))
    y_train_batch_01 = np.int64(np.zeros((batches)))

    Movements_predicted = 0.0
    for t in range(0, T_eval, batches):
        for k in range(batches):
            kk =random_index_list[k]
            x_test_batch_1[k, :, :] = np.transpose(X_np_eval_1[t+kk :t+kk + BPTT_length, :])
            y_test_batch_1[k] = Y_np_eval_1[t+kk + BPTT_length-1]
            end_idx_01 = (t+kk + BPTT_length-1)*time_interval_ratio
            x_test_batch_01[k, :, :] = np.transpose(X_np_eval_01[end_idx_01-BPTT_length+1 :end_idx_01+1, :])
            y_test_batch_01[k] = Y_np_eval_01[end_idx_01]
        actual_out, probs_1 = CNN_Model(stock_name, x_test_batch_1, y_test_batch_1, x_test_batch_01, y_test_batch_01)
        #add LSTM model
        # probs = (probs_1+probs_2)/2.0
        probs = probs_1
        probs[:, 1] = 0
        pred_out = [np.argmax(probs, 1)]
        pred_out = np.array(pred_out).reshape(-1)
        counter += 1
        Movements_predicted += sum([1 for p, j in zip(actual_out, pred_out) if p != 1 and (p == j)])
        total_movements += sum([1 for p, j in zip(actual_out, pred_out) if p!= 1])

    conditional_accuracy = np.array(Movements_predicted) / total_movements
    print(Movements_predicted, 'Movements_predicted correctly')
    print(total_movements, 'Total Movements')
    print(conditional_accuracy, 'Conditional Accuracy')
    cond_acc[i] = conditional_accuracy

cond_acc = pd.DataFrame(cond_acc)
cond_acc.to_csv(store_path+stock_name'_cond_acc.csv')
time2 = time.time()
print(time2-time1)
