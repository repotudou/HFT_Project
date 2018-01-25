import numpy as np
import os
import sys
import h5py
import time

def _check_existence(stock_ticker, time_interval,folder_1 = './',folder_01 = './'):
    
    if time_interval == 1:
        folder = folder_1
        stock_file_name = stock_ticker + '_loadRNN_1.hdf5'
    elif time_interval ==0.1:
        folder = folder_01
        stock_file_name = stock_ticker + '_loadRNN_01.hdf5'
    else:
        print('Wrong time interval.')
        return False
    file_name_list = os.listdir(folder)
    # folder =''
    exist_bool = False
    # file_name_list = ['AMD_loadRNN_1.hdf5']
   # print(stock_ticker)
    for item in file_name_list:
        if item == stock_file_name:
            print("Find it")
            exist_bool = True
    return exist_bool



def _load_train_test_data(stock_ticker, time_interval, folder_1 = './',folder_01 = './'):

    if not _check_existence(stock_ticker, time_interval, folder_1 = folder_1, folder_01 = folder_01):
        print("There is an error in stock %s in time interval %f"%(stock_ticker,time_interval))
        exit()
    else:
        if time_interval == 1:
            folder = folder_1
            stock_file_name = stock_ticker + '_loadRNN_1.hdf5'
        elif time_interval ==0.1:
            folder = folder_01
            stock_file_name = stock_ticker + '_loadRNN_01.hdf5'
        else:
            print('Wrong time interval.')
            return False
        HDF5_file_LoadRNN = h5py.File(folder + stock_file_name, 'r')
    return HDF5_file_LoadRNN


def _trainAndtest_length(stock_ticker, time_interval,  folder_1 = './', folder_01 = './'):
    HDF5_file_LoadRNN = _load_train_test_data(stock_ticker,time_interval, folder_1 = folder_1, folder_01 = folder_01)   
    X_train= HDF5_file_LoadRNN['X_train']
    y_train = HDF5_file_LoadRNN['y_train']
    X_test = HDF5_file_LoadRNN['X_test']
    y_test = HDF5_file_LoadRNN['y_test']
    return y_train.len(),y_test.len()



if __name__ == '__main__':
    stock_ticker = 'AMD'
    time_interval_1 = 1
    time_interval_01 = 0.1
    folder_1 = './'
    folder_01 = './'
    HDF5_file_LoadRNN_1 = _load_train_test_data(stock_ticker, time_interval_1, folder_1 = folder_1, folder_01 = folder_01)
    X_train_1 = HDF5_file_LoadRNN_1['X_train']
    X_test_1 = HDF5_file_LoadRNN_1['X_test']
    y_train_1 = HDF5_file_LoadRNN_1['y_train']
    y_test_1 = HDF5_file_LoadRNN_1['y_test']
   
    HDF5_file_LoadRNN_01 = _load_train_test_data(stock_ticker, time_interval_01,  folder_1 = folder_1, folder_01 = folder_01)
    X_train_01 = HDF5_file_LoadRNN_01['X_train']
    X_test_01 = HDF5_file_LoadRNN_01['X_test']
    y_train_01 = HDF5_file_LoadRNN_01['y_train']
    y_test_01 = HDF5_file_LoadRNN_01['y_test']
   
    train_len_1, test_len_1 = _trainAndtest_length(stock_ticker, time_interval_1,  folder_1 = folder_1, folder_01 = folder_01)
    train_len_01, test_len_01 = _trainAndtest_length(stock_ticker, time_interval_01,  folder_1 = folder_1, folder_01 = folder_01)
#**************************************************************************************************************    
    num_per_day = 20000
    batch_size = 1024

    epoch_limit = 10
    BPTT_length = 20
    num_inputs = 24

    LSTM_lasting_time = 20*BPTT_length
    num_BPTT_iterations = int(LSTM_lasting_time/BPTT_length)
    time_interval = 1

    time_interval_ratio = int(time_interval_1/time_interval_01)
#**************************************************************************************************************    
    train_num_of_days = int(train_len_1/num_per_day)

    print(train_num_of_days)



    for ii in range(epoch_limit):

        random_day = np.random.choice(train_num_of_days, batch_size)
        random_day_index = random_day * num_per_day
        print(random_day_index)
        random_start_time = np.random.choice(np.arange(0, num_per_day - LSTM_lasting_time), batch_size)
        print(random_start_time)
        random_start_time_index = random_day_index + random_start_time
        print(random_start_time_index)


        X_train_batch_1 = np.float32(np.zeros((batch_size,BPTT_length,num_inputs)))
        y_train_batch_1 = np.int32(np.zeros((batch_size,BPTT_length)))
        X_train_batch_01 = np.float32(np.zeros((batch_size,BPTT_length,num_inputs)))
        y_train_batch_01 = np.int32(np.zeros((batch_size,BPTT_length)))

        for i in range(num_BPTT_iterations):
            print(random_start_time_index)
            for k in range(batch_size):
                batch_k_start_index_1 = random_start_time_index[k]
                batch_k_end_index_1 = batch_k_start_index_1 + BPTT_length 
                
                X_train_batch_1[k,:,:] = X_train_1[batch_k_start_index_1: batch_k_end_index_1,:]
                y_train_batch_1[k,:] = np.array(y_train_1[batch_k_start_index_1: batch_k_end_index_1]).reshape(BPTT_length)
        
                batch_k_end_index_01 = (batch_k_end_index_1 - 1) * time_interval_ratio + 1 
                batch_k_start_index_01 = batch_k_end_index_01 - BPTT_length
                X_train_batch_01[k,:,:] = X_train_01[batch_k_start_index_01: batch_k_end_index_01,:]
                y_train_batch_01[k,:] = np.array(y_train_01[batch_k_start_index_01: batch_k_end_index_01]).reshape(BPTT_length)
            random_start_time_index = random_start_time_index + BPTT_length 
           
            for k in range(batch_size):

                if not np.array_equal(X_train_batch_1[k,-1,:-1],X_train_batch_01[k,-1,:-1]):
                    print(X_train_batch_1[k,-1,:-1])
                    print(X_train_batch_01[k,-1,:-1])
                    print("not equal")