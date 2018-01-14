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
epoch_limit=60
keep_prob_train_1=0.95
keep_prob_train_01=0.95
T = 1000*batches
T_eval = 100*batches
levels = 10
time_interval_ratio = 10

folder = '/home/leifan/Data/1Y/20Stocks_LoadRNNdata_1/'
# folder = '/Users/leifan/Dropbox/VDCNN/'
stock_name = 'HRL'
stock_file_name = stock_name+'_loadRNN_1.hdf5'

model_identifier = stock_name
HDF5_file = h5py.File(folder + stock_file_name, 'r')
X_np_train_1 = HDF5_file['X_train']
Y_np_train_1 = HDF5_file['y_train']
X_np_eval_1 = HDF5_file['X_test']
Y_np_eval_1 = HDF5_file['y_test']


folder_01 = '/home/leifan/Data/1Y/20Stocks_LoadRNNdata_01/'
# folder = '/Users/leifan/Dropbox/VDCNN/'
stock_file_name_01 = stock_name+'_loadRNN_01.hdf5'
HDF5_file_01 = h5py.File(folder_01 + stock_file_name_01, 'r')
X_np_train_01 = HDF5_file_01['X_train']
Y_np_train_01 = HDF5_file_01['y_train']
X_np_eval_01 = HDF5_file_01['X_test']
Y_np_eval_01 = HDF5_file_01['y_test']


# num_train_1,_ = X_np_train_1.shape
# # select_idx = range(0,num_train_01,10)
# counter = 0
# l =[]
# for i in range(num_train_1):
#     print('num of epoch', i+1)
#     if np.array_equal(X_np_train_1[i,0:23],X_np_train_01[i*10,0:23]):
#         pass
#     else:
#         print("Warning!!!!")
#         counter +=1
#         l.append(i)
# print(counter/np.float32(num_train_1))
# print(l)


# print(X_np_train_01[select_idx,:]==X_np_train_1)
# print(Y_np_train_01[select_idx,:]==Y_np_train_1)

# print(X_np_train_1.shape)
# print(X_np_train_01.shape)
# print(Y_np_train_1.shape)
# print(Y_np_train_01.shape)
# print(X_np_eval_1.shape)
# print(X_np_eval_01.shape)
# print(Y_np_eval_1.shape)
# print(Y_np_eval_01.shape)



# idx = [i for i in range(X_np_train_1.shape[0]) if Y_np_train_1[i]==2]
# print(idx)

training_case = [LR, num_rep_block, time_lenght,pooling_filter_size,cnn_filter_size,num_filters_per_size, num_levels, epsilonADAM, keep_prob_train_1, T, T_eval, batches, stock_name]
HP = '_' + 'num_rep_block' + str(num_rep_block) + '_' + 'batches'+ str(batches) + '_' + 'time_lenght' + str(time_lenght) +'_' + 'Dropout' + str(keep_prob_train_1)

# [0.001, (2, 2, 2, 2), 40, 2, 3, (64, 128, 256, 512), 4, 1e-08, 0.95, 100000, 4000, 100, 'AMD']
# _num_rep_block(2, 2, 2, 2)_batches100_time_lenght40_Dropout0.95

def resUnit(input_layer, num_filters_per_size_i, cnn_filter_size, i, j):
    with tf.variable_scope("res_unit_" + str(i) + "_" + str(j)):
        part1 = slim.conv2d(input_layer, num_filters_per_size_i, [1, cnn_filter_size], activation_fn=None)
        part2 = slim.batch_norm(part1, activation_fn=None)
        part3 = tf.nn.relu(part2)
        part4 = slim.conv2d(part3, num_filters_per_size_i, [1, cnn_filter_size], activation_fn=None)
        part5 = slim.batch_norm(part4, activation_fn=None)
        part6 = tf.nn.relu(part5)
        output = part6
        return output

input_x1_1 =  tf.placeholder(tf.float32, shape=[None,num_inputs,time_lenght],name="input_x_1")
input_x_1=tf.reshape(input_x1_1,[-1,num_inputs,time_lenght,1])
input_y_1 = tf.placeholder(tf.int64, shape=[None],name="input_y_1")
dropout_keep_prob_1 = tf.placeholder(tf.float32, name="dropout_keep_prob_1")


input_x1_01 =  tf.placeholder(tf.float32, shape=[None,num_inputs,time_lenght],name="input_x_01")
input_x_01=tf.reshape(input_x1_01,[-1,num_inputs,time_lenght,1])
input_y_01 = tf.placeholder(tf.int64, shape=[None],name="input_y_01")
dropout_keep_prob_01 = tf.placeholder(tf.float32, name="dropout_keep_prob_01")



h_1 = slim.conv2d(input_x_1, num_filters_per_size[0], [num_inputs, cnn_filter_size], normalizer_fn=slim.batch_norm, scope = 'conv0_1', padding='VALID')
h_01 = slim.conv2d(input_x_01, num_filters_per_size[0], [num_inputs, cnn_filter_size], normalizer_fn=slim.batch_norm, scope = 'conv0_01', padding='VALID')
# ================ Conv Block 64, 128, 256, 512 ================

for i in range(0,len(num_filters_per_size)):
    for j in range(0,num_rep_block[i]):
        h_1 = resUnit(h_1, num_filters_per_size[i], cnn_filter_size, i, j)
    h_1 = slim.max_pool2d(h_1, [1,pooling_filter_size], scope='pool_1_%s' % i)
# ================ Layer FC ================

# ================ Conv Block 64, 128, 256, 512 ================

for i in range(0,len(num_filters_per_size)):
    for j in range(0,num_rep_block[i]):
        h_01 = resUnit(h_01, num_filters_per_size[i], cnn_filter_size, i, j)
    h_01 = slim.max_pool2d(h_01, [1,pooling_filter_size], scope='pool_01_%s' % i)
# ================ Layer FC ================

# Global avg max pooling
h_1 = math_ops.reduce_mean(h_1, [1, 2], name='pool5_1', keep_dims=True)
h_01 = math_ops.reduce_mean(h_01, [1, 2], name='pool5_01', keep_dims=True)
# Conv
h_1 = slim.conv2d(h_1, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='output_1')
h_01 = slim.conv2d(h_01, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='output_01')


# FC & Dropout
scores_1 = slim.flatten(h_1)
scores_01 = slim.flatten(h_01)

#fully connected layer
output = tf.concat(1, [scores_1, scores_01])
dense = tf.layers.dense(inputs=output, units = 30, activation = tf.nn.relu)
scores = tf.layers.dense(inputs=dense, units=3)

u_prob=tf.nn.softmax(scores)
pred1D = tf.argmax(scores, 1, name="predictions")
# print(pred1D,'pred1')

# ================ Loss and Accuracy ================
# CalculateMean cross-entropy loss
with tf.name_scope("evaluate"):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores,labels=input_y_01)
    loss = tf.reduce_mean(losses)
    correct_predictions = tf.equal(pred1D, input_y_01)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

optimizer = tf.train.AdamOptimizer(learning_rate = LR, beta1 = 0.9, beta2 = .999, epsilon = epsilonADAM)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()


session = tf.Session()
session.run(init)


#save path
header = '/home/leifan/Dropbox/Test_CNN/'
# os.system('mkdir'+ ' '+header+stock_name+'_01')
os.system('mkdir'+ ' '+header+stock_name+'_Dual')
path = header+stock_name+'_Dual/'
os.system('mkdir'+ ' '+path+'save')
save_path = path+'save/'
path_saver = save_path + "Test" + model_identifier + HP + ".ckpt"

#save model
saver = tf.train.Saver()
# saver.restore(session, path_saver)


eval_error_tracker = []
train_error_tracker = []
train_err = np.zeros(epoch_limit)
test_err = np.zeros(epoch_limit)
train_acc = np.zeros(epoch_limit)
test_acc = np.zeros(epoch_limit)
cond_acc = np.zeros(epoch_limit)

for i in range(epoch_limit):
    print('epoch: %d' %(i+1))
    current_train_error = 0.0
    current_train_acc = 0.0
    # TRAIN MODEL
    random_index_list = []
    for k in range(batches):
        random_index = random.randint(0,len(Y_np_train_1)-T-1)
        random_index_list.append(random_index)
    counter = 0
    train_error_avg=0
    x_train_batch_1 = np.float32(np.zeros((batches, num_inputs,time_lenght)))
    y_train_batch_1 = np.int64(np.zeros((batches)))
    x_train_batch_01 = np.float32(np.zeros((batches, num_inputs,time_lenght)))
    y_train_batch_01 = np.int64(np.zeros((batches)))

    for t in range(0, T, batches):
        for k in range(batches):
            kk =random_index_list[k]
            x_train_batch_1[k, :, :] = np.transpose(X_np_train_1[t+kk :t+kk + time_lenght, :])
            y_train_batch_1[k] = Y_np_train_1[t+kk + time_lenght-1]
            end_idx_01 = (t+kk + time_lenght-1)*time_interval_ratio
            x_train_batch_01[k, :, :] = np.transpose(X_np_train_01[end_idx_01-time_lenght+1 :end_idx_01+1, :])
            y_train_batch_01[k] = Y_np_train_01[end_idx_01]

            # print(x_train_batch_1)
            # print(y_train_batch_01)
        # print(y_train_batch_1==y_train_batch_01)
        _,train_accuracy, loss_np = session.run([train,accuracy, loss], 
            feed_dict={input_x1_1: x_train_batch_1, input_y_1: y_train_batch_1, dropout_keep_prob_1: keep_prob_train_1,
                        input_x1_01: x_train_batch_01, input_y_01: y_train_batch_01, dropout_keep_prob_01: keep_prob_train_01})
        current_train_error += loss_np
        current_train_acc += train_accuracy
        counter += 1
    train_loss_avg=current_train_error / np.float(counter)
    train_error_tracker.append(train_loss_avg)
    train_err[i] = train_loss_avg
    train_acc[i] = current_train_acc / np.float(counter)
    # EVALUATION ERROR
    current_eval_error = 0.0
    current_eval_acc = 0.0
    random_index_list = []
    total_movements=1.0

    for k in range(batches):
        random_index = random.randint(0,len(Y_np_eval_1)-T_eval-1)
        random_index_list.append(random_index)
    counter = 0
    x_test_batch_1 = np.float32(np.zeros((batches, num_inputs,time_lenght)))
    y_test_batch_1 = np.int64(np.zeros((batches)))
    x_test_batch_01 = np.float32(np.zeros((batches, num_inputs,time_lenght)))
    y_test_batch_01 = np.int64(np.zeros((batches)))

    Movements_predicted = 0.0

    for t in range(0, T_eval, batches):
        for k in range(batches):
            kk =random_index_list[k]
            x_test_batch_1[k, :, :] = np.transpose(X_np_eval_1[t+kk :t+kk + time_lenght, :])
            y_test_batch_1[k] = Y_np_eval_1[t+kk + time_lenght-1]
            end_idx_01 = (t+kk + time_lenght-1)*time_interval_ratio
            x_test_batch_01[k, :, :] = np.transpose(X_np_eval_01[end_idx_01-time_lenght+1 :end_idx_01+1, :])
            y_test_batch_01[k] = Y_np_eval_01[end_idx_01]
        # print(y_test_batch_1==y_test_batch_01)

        loss_np, accuracy_np,actual_out,probs = session.run([loss, accuracy,input_y_01,u_prob],
            feed_dict={input_x1_1: x_test_batch_1, input_y_1: y_test_batch_1,dropout_keep_prob_1: np.float32(1.0),
                        input_x1_01: x_test_batch_01, input_y_01: y_test_batch_01, dropout_keep_prob_01: np.float32(1.0)})
        probs[:, 1] = 0
        pred_out = [np.argmax(probs, 1)]
        pred_out = np.array(pred_out).reshape(-1)
        current_eval_error += loss_np
        current_eval_acc += accuracy_np
        counter += 1
        Movements_predicted += sum([1 for p, j in zip(actual_out, pred_out) if p != 1 and (p == j)])
        total_movements += sum([1 for p, j in zip(actual_out, pred_out) if p!= 1])

    conditional_accuracy = np.array(Movements_predicted) / total_movements
    print(Movements_predicted, 'Movements_predicted correctly')
    print(total_movements, 'Total Movements')
    print(conditional_accuracy, 'Conditional Accuracy')
    accuracy_np_ave = current_eval_acc / np.float32(counter)
    loss_np_ave = current_eval_error / np.float32(counter)
    test_acc[i] = accuracy_np_ave
    test_err[i] = loss_np_ave
    cond_acc[i] = conditional_accuracy
    eval_error_tracker.append((accuracy_np_ave, loss_np_ave, conditional_accuracy))
    print(model_identifier, 'train_err', train_err[i],  'test_err', test_err[i], 'train_acc', train_acc[i], 'test_acc', test_acc[i], 'conditional_acc', cond_acc[i])
    if (i+1)%5 == 0:
        saver.save(session, path_saver)
#     # path_saver = "/Users/leifan/Reserach/VDCNN/" + "Test" + model_identifier + HP + ".ckpt"
#     # save_path = saver.save(session, path_saver)
#     # pickle_list0[0] = eval_error_tracker
#     # pickle_list0[1] = train_error_tracker
#     # pickle.dump(pickle_list0, open('/Users/leifan/Reserach/VDCNN/' + "Test" + model_identifier + HP + '.p','wb'))
train_err = pd.DataFrame(train_err)
train_err.to_csv(path+'train_err.csv')
train_acc = pd.DataFrame(train_acc)
train_acc.to_csv(path+'train_acc.csv')
test_err = pd.DataFrame(test_err)
test_err.to_csv(path+'test_err.csv')
test_acc = pd.DataFrame(test_acc)
test_acc.to_csv(path+'test_acc.csv')
cond_acc = pd.DataFrame(cond_acc)
cond_acc.to_csv(path+'cond_acc.csv')
time2 = time.time()
print(time2-time1)
