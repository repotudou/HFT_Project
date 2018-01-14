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


def load_train_test_data(stock_ticker, time_interval):
    if time_interval == 1:
         folder = '/projects/sciteam/bahp/DeepLearningHF/20Stocks_LoadRNNData_1year/20Stocks_LoadRNNdata_1/'
    elif time_interval ==0.1:
         folder = '/projects/sciteam/bahp/DeepLearningHF/20Stocks_LoadRNNData_1year/20Stocks_LoadRNNdata_01/'
    else:
         print('Wrong time interval.')
         return False
    file_name_list = os.listdir(folder)
    # folder =''
    exist_bool = False
    stock_file_name =''
    #file_name_list = ['AMD_loadRNN_1.hdf5','BIDU_loadRNN_1.hdf5']
    # print(stock_ticker)
    for item in file_name_list:
        if len(item.split('.')) == 2:
            if item.split('.')[1] =='hdf5' and item.split('_')[0] == stock_ticker:
                stock_file_name = item
                exist_bool = True
                print('Find it')
                break
            else:
                pass
    if exist_bool==False:
        return [exist_bool]
    print(stock_file_name)
    HDF5_file_LoadRNN = h5py.File(folder +stock_file_name, 'r')
    return [exist_bool,HDF5_file_LoadRNN]

def resUnit(input_layer, num_filters_per_size_i, cnn_filter_size, i, j):
    with tf.variable_scope("res_unit_" + str(i) + "_" + str(j)):
        part1 = slim.batch_norm(input_layer, activation_fn=None)
        part2 = tf.nn.relu(part1)
        part3 = slim.conv2d(part2, num_filters_per_size_i, [1, cnn_filter_size], activation_fn=None)
        part4 = slim.batch_norm(part3, activation_fn=None)
        part5 = tf.nn.relu(part4)
        part6 = slim.conv2d(part5, num_filters_per_size_i, [1, cnn_filter_size], activation_fn=None)
        output = part6
        return output


class CNN_Model():
	def __init__(self, config):
		self.num_inputs = config['num_inputs']
		self.batch_size = config['batch_size']
		filter_size = config['filter_size']
		num_rep_block = config['num_rep_block']
		num_filters_per_size = config['num_filters_per_size']
		self.BPTT_length = config['BPTT_length']
		learning_rate = config['learning_rate']
		beta1 = config['beta1']
		beta2 = config['beta2']
		epsilonADAM = config['epsilonADAM']
		num_classes = config['num_classes']
		summary_bool = config['summary_bool']
		pooling_filter_size = config['pooling_filter_size']
		self.input_x1 =  tf.placeholder(tf.float32, shape=[None,self.num_inputs,self.BPTT_length],name="input_x")
		self.input_x=tf.reshape(self.input_x1,[-1,self.num_inputs,self.BPTT_length,1])
		self.input_y = tf.placeholder(tf.int64, shape=[None],name="input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")


		#first CNN layer
		h = slim.conv2d(self.input_x, num_filters_per_size[0], [self.num_inputs, filter_size], normalizer_fn=slim.batch_norm, scope = 'conv0', padding='VALID')
		# ================ Conv Block 64, 128, 256, 512 ================

		for i in range(0,len(num_filters_per_size)):
		    for j in range(0,num_rep_block[i]):
		        h = resUnit(h, num_filters_per_size[i], filter_size, i, j)
		    h = slim.max_pool2d(h, [1, pooling_filter_size], scope='pool_%s' % i)
		# ================ Layer FC ================
		# Global avg max pooling
		h = math_ops.reduce_mean(h, [1, 2], name='pool5', keep_dims=True)

		# Conv
		h = slim.conv2d(h, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='output')

		# FC & Dropout
		scores = slim.flatten(h)
		self.u_prob=tf.nn.softmax(scores)
		pred1D = tf.argmax(scores, 1, name="predictions")

		# ================ Loss and Accuracy ================
		# CalculateMean cross-entropy loss
		with tf.name_scope("evaluate"):
			losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores,labels=self.input_y)
			self.loss = tf.reduce_mean(losses)
			correct_predictions = tf.equal(pred1D, self.input_y)
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
			
		with tf.name_scope("Optimizer"):
			optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = beta1, beta2 = beta2, epsilon = epsilonADAM)
			self.train = optimizer.minimize(self.loss)

		self.init_op = tf.global_variables_initializer()
		print('Finished computation graph')

# test CNN_model.py
# if __name__ == '__main__':
    
#     stock_ticker = 'AMD'
#     time_interval = 1
#     x = load_train_test_data(stock_ticker, time_interval)
#     print(x[1])
