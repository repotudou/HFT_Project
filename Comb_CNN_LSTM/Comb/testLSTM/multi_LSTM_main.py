import numpy as np
import pandas as pd
import tensorflow as tf  
from multi_LSTM_model import Multi_LSTM_Model
from load_data import _load_train_test_data, _trainAndtest_length

# from mpi4py import MPI
import os
import time


# node_name  = MPI.Get_processor_name()
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
    # rank = 0
    # name = 'hello'

#*********************************************
# fixed variables    
time_interval_1 = 1
time_interval_01 = 0.1
time_interval_ratio = int(time_interval_1/time_interval_01)
num_classes = 3
num_levels = 10
num_inputs = int(num_levels*2+4)
num_per_day = 20000
# Hyperparamaters
num_layers = 2
hidden_size = 64
BPTT_length = 20
batch_size = 1024
LSTM_lasting_time = 20*BPTT_length
train_dropout_1 = 0.8
train_dropout_01 = 0.8
test_dropout = 1.0
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
epsilonADAM = 1e-8
#*****************************************
epoch_limit = 5



# stock_file_list = [ 'AMD','BIDU', 'RCL', 'CBS', 'D', 'SYMC', 'EMR', 'ENDP', 'UPS', 'HON', 'URBN', 'HRL', 'VFC', 'JNPR', 'VIAB', 'JWN', 'KMB']
#stock_file_list = ['RCL','CBS']
# stock_ticker = stock_file_list[rank]
stock_ticker = 'AMD'
folder_1 = '/home/leifan/Data/1Y/20Stocks_LoadRNNdata_1/'
folder_01 = '/home/leifan/Data/1Y/20Stocks_LoadRNNdata_01/'
# folder_01 ='./'
# folder_1 ='./'

start_time = time.time()

HDF5_file_LoadRNN_1 = _load_train_test_data(stock_ticker, time_interval_1, folder_1 = folder_1, folder_01 =folder_01)
X_train_1 = HDF5_file_LoadRNN_1['X_train']
X_test_1 = HDF5_file_LoadRNN_1['X_test']
y_train_1 = HDF5_file_LoadRNN_1['y_train']
y_test_1 = HDF5_file_LoadRNN_1['y_test']

HDF5_file_LoadRNN_01 = _load_train_test_data(stock_ticker, time_interval_01, folder_1 = folder_1, folder_01 =folder_01)
X_train_01 = HDF5_file_LoadRNN_01['X_train']
X_test_01 = HDF5_file_LoadRNN_01['X_test']
y_train_01 = HDF5_file_LoadRNN_01['y_train']
y_test_01 = HDF5_file_LoadRNN_01['y_test']

train_len_1, test_len_1 = _trainAndtest_length(stock_ticker, time_interval_1, folder_1 = folder_1, folder_01 =folder_01)
train_len_01, test_len_01 = _trainAndtest_length(stock_ticker, time_interval_01, folder_1 = folder_1, folder_01 =folder_01)
num_BPTT_iterations = int(LSTM_lasting_time/BPTT_length)
train_num_of_days = int(train_len_1/num_per_day)
test_num_of_days = int(test_len_1/num_per_day)

config = {    'num_layers' :    num_layers,               #number of layers of stacked RNN's
              'hidden_size' :   hidden_size,             #memory cells in a layer
              'BPTT_length' :   BPTT_length,
              'batch_size' :    batch_size,
              'num_inputs':     num_inputs ,           
              'learning_rate' : learning_rate,
              'beta1'  :        beta1,  
              'beta2'  :        beta2, 
              'epsilonADAM' :   epsilonADAM,          
              'num_classes':    num_classes,
              'time_interval_ratio': time_interval_ratio}

model = Multi_LSTM_Model(config)

sess = tf.Session()
sess.run(model.init_op)
saver = tf.train.Saver()
file_path = '/home/leifan/Dropbox/Comb/MultiLSTM/AMD/'
save_path = file_path+'save/'
path_saver = save_path + "MultiLSTM.ckpt"



saver.restore(sess, path_saver)
#define savable values

cond_acc = np.zeros(epoch_limit)

for ii in range(epoch_limit):
    print('epoch: %d' %(ii+1))
    #*****************************************************************************************************************               
    # Testing 
    counter = 0
    test_state_1 = sess.run(model.reset_state_1)

    test_random_day = np.random.choice(test_num_of_days, batch_size)
    test_random_day_index = test_random_day * num_per_day
    test_random_start_time = np.random.choice(np.arange(0, num_per_day - LSTM_lasting_time), batch_size)
    test_random_start_time_index = test_random_day_index + test_random_start_time


    X_test_batch_1 = np.float32(np.zeros((batch_size,BPTT_length,num_inputs)))
    y_test_batch_1 = np.int32(np.zeros((batch_size,BPTT_length)))
    X_test_batch_01 = np.float32(np.zeros((batch_size,BPTT_length,num_inputs)))
    y_test_batch_01 = np.int32(np.zeros((batch_size,BPTT_length)))

    total_movements = 0
    correct_movements = 0
    
    for i in range(num_BPTT_iterations):
        for k in range(batch_size):    
            batch_k_start_index_1 = test_random_start_time_index[k]
            batch_k_end_index_1 = batch_k_start_index_1 + BPTT_length 
            
            X_test_batch_1[k,:,:] = X_test_1[batch_k_start_index_1: batch_k_end_index_1,:]
            y_test_batch_1[k,:] = np.array(y_test_1[batch_k_start_index_1: batch_k_end_index_1]).reshape(BPTT_length)

            batch_k_end_index_01 = (batch_k_end_index_1-1) * time_interval_ratio + 1
            batch_k_start_index_01 = batch_k_end_index_01 - BPTT_length
            
            X_test_batch_01[k,:,:] = X_test_01[batch_k_start_index_01: batch_k_end_index_01,:]
            y_test_batch_01[k,:] = np.array(y_test_01[batch_k_start_index_01: batch_k_end_index_01]).reshape(BPTT_length)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 
        test_state_01 = sess.run(model.reset_state_01)
        cost_test, acc_test,prob, test_state_1,_ = sess.run([model.cost, model.accuracy, model.logits, model.next_state_1,model.next_state_01],
                feed_dict = {model.input_1: X_test_batch_1, model.labels_1: y_test_batch_1,
                            model.input_01: X_test_batch_01, model.labels_01: y_test_batch_01,
                            model.keep_prob_1:test_dropout, model.keep_prob_01:test_dropout,
                            model.state_1:test_state_1, model.state_01:test_state_01})

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        test_random_start_time_index = test_random_start_time_index + BPTT_length

        counter += 1
        
        y_test_batch = y_test_batch_01

        for l in range(batch_size):
            if y_test_batch[l,-1] != 1:
                total_movements += 1
                conditional_pred = None
                if (prob[l,0]>=prob[l,2]):
                    conditional_pred = 0
                else:
                    conditional_pred = 2
                if conditional_pred == y_test_batch[l,-1]:
                    correct_movements += 1
    conditional_acc = correct_movements/float(total_movements)
    print(conditional_acc, 'Conditional Accuracy')
    cond_acc[ii] = conditional_acc
    # if (ii+1)%10==0:
    #     saver.save(sess, path_saver)


cond_acc = pd.DataFrame(cond_acc)
cond_acc.to_csv(file_path+'cond_acc.csv')
end_time = time.time()
print(end_time-start_time)
