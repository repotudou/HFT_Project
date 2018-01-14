import numpy as np
import tensorflow as tf
import time
import copy
import os
import sys
import h5py
import random
import pandas as pd
import pickle
from CNN_1 import CNN_1
from CNN_2 import CNN_2
from CNN_3 import CNN_3
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import math_ops
from tensorflow.contrib import layers as layers_lib

time1 = time.time()
LR = .001
epsilonADAM = 1e-8
time_lenght=80
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
keep_prob_train=0.95
T = 1000*batches
T_eval = 1*batches
levels = 10


folder = '/home/leifan/Data/1Y/20Stocks_LoadRNNdata_1/'
# folder = '/home/leifan/Data/20Stocks_LoadRNNdata_1/'
stock_file_name = 'AMD_loadRNN_1.hdf5'
stock_name = 'AMD'
model_identifier = stock_name
HDF5_file = h5py.File(folder + stock_file_name, 'r')
X_np_eval = HDF5_file['X_test']
Y_np_eval = HDF5_file['y_test']


# eval_error_tracker = []
# train_error_tracker = []
# train_err = np.zeros(epoch_limit)
# test_err = np.zeros(epoch_limit)
# train_acc = np.zeros(epoch_limit)
# test_acc = np.zeros(epoch_limit)
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
    x_batch = np.float32(np.zeros((batches, num_inputs,time_lenght)))
    y_batch = np.int32(np.zeros((batches)))
    Movements_predicted = 0.0
    for t in range(0, T_eval, batches):
        for k in range(batches):
            kk =random_index_list[k]
            x_batch[k, :, :] = np.transpose(X_np_eval[t+kk :t+kk + time_lenght, :])
            y_batch[k] = Y_np_eval[t+kk + time_lenght-1]
        actual_out, probs_1 = CNN_1('AMD', x_batch, y_batch)
        actual_out, probs_2 = CNN_2('AMD', x_batch, y_batch)
        actual_out, probs_3 = CNN_3('AMD', x_batch, y_batch)
        probs_1[:, 1] = 0
        probs_2[:, 1] = 0
        probs_3[:, 1] = 0
        pred_out_1 = [np.argmax(probs_1, 1)]
        pred_out_1 = np.array(pred_out_1).reshape(-1)
        pred_out_2 = [np.argmax(probs_2, 1)]
        pred_out_2 = np.array(pred_out_2).reshape(-1)
        pred_out_3 = [np.argmax(probs_3, 1)]
        pred_out_3 = np.array(pred_out_3).reshape(-1)
        pred_out = pred_out_1[:]
        pred_out = np.array(pred_out).reshape(-1)
        actual_out_1 = np.array(actual_out).reshape(-1)
        temp1 = np.vstack((pred_out_1,pred_out_2))
        temp2 = np.vstack((pred_out_3,pred_out))
        temp = np.vstack((temp1, temp2))
        temp = np.vstack((temp, actual_out_1))
        temp = np.transpose(temp)
        temp = pd.DataFrame(temp)
        temp.to_csv('temp.csv')
        # print(pred_out_1==pred_out_2)
        # print(pred_out_2==pred_out_3)
        # print(pred_out_3==pred_out_1)
        # print(pred_out)
        # print(len(pred_out))
        for l in range(len(pred_out_1)):
            if pred_out_1[l]==pred_out_2[l]:
                pred_out[l] = pred_out_1[l]
            else:
                if pred_out_3[l]==pred_out_1[l]:
                    pred_out[l] = pred_out_1[l]
                else:
                    pred_out[l] = pred_out_2[l]

        pred_out = np.array(pred_out).reshape(-1)
        counter += 1
        Movements_predicted += sum([1 for p, j in zip(actual_out, pred_out) if p != 1 and (p == j)])
        total_movements += sum([1 for p, j in zip(actual_out, pred_out) if p!= 1])

    conditional_accuracy = np.array(Movements_predicted) / total_movements
    print(Movements_predicted, 'Movements_predicted correctly')
    print(total_movements, 'Total Movements')
    print(conditional_accuracy, 'Conditional Accuracy')
    # accuracy_np_ave = current_eval_acc / np.float32(counter)
    # loss_np_ave = current_eval_error / np.float32(counter)
    # test_acc[i] = accuracy_np_ave
    # test_err[i] = loss_np_ave
    cond_acc[i] = conditional_accuracy
    # eval_error_tracker.append((accuracy_np_ave, loss_np_ave, conditional_accuracy))
    # print(model_identifier, 'train_err', train_err[i],  'test_err', test_err[i], 'train_acc', train_acc[i], 'test_acc', test_acc[i], 'conditional_acc', cond_acc[i])
    # if (i+1)%5 == 0:
    #     saver.save(session, path_saver)
#     # path_saver = "/Users/leifan/Reserach/VDCNN/" + "Test" + model_identifier + HP + ".ckpt"
#     # save_path = saver.save(session, path_saver)
#     # pickle_list0[0] = eval_error_tracker
#     # pickle_list0[1] = train_error_tracker
#     # pickle.dump(pickle_list0, open('/Users/leifan/Reserach/VDCNN/' + "Test" + model_identifier + HP + '.p','wb'))
# train_err = pd.DataFrame(train_err)
# train_err.to_csv('train_err.csv')
# train_acc = pd.DataFrame(train_acc)
# train_acc.to_csv('train_acc.csv')
# test_err = pd.DataFrame(test_err)
# test_err.to_csv('test_err.csv')
# test_acc = pd.DataFrame(test_acc)
# test_acc.to_csv('test_acc.csv')
cond_acc = pd.DataFrame(cond_acc)
cond_acc.to_csv('cond_acc.csv')
time2 = time.time()
print(time2-time1)



