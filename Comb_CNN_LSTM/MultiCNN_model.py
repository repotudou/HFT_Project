import numpy as np
import tensorflow as tf 
import time
import os
import sys
import random
import pandas as pd
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import math_ops
from tensorflow.contrib import layers as layers_lib


def resUnit(input_layer, num_filters_per_size_i, cnn_filter_size, i, j):
    with tf.variable_scope("res_unit_" + str(i) + "_" + str(j)):
        part1 = slim.batch_norm(input_layer, activation_fn=None)
        part2 = tf.nn.relu(part1)
        part3 = slim.conv2d(part2, num_filters_per_size_i, [cnn_filter_size, 1], activation_fn=None)
        part4 = slim.batch_norm(part3, activation_fn=None)
        part5 = tf.nn.relu(part4)
        part6 = slim.conv2d(part5, num_filters_per_size_i, [cnn_filter_size, 1], activation_fn=None)
        output = part6
        return output

class MultiCNN_Model():
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
		pooling_filter_size = config['pooling_filter_size']

		self.input_1 = tf.placeholder(tf.float32,[None,self.BPTT_length,self.num_inputs],name ='input_1')
		input_1=tf.reshape(self.input_1,[-1,self.BPTT_length,self.num_inputs,1])
		self.input_01 = tf.placeholder(tf.float32,[None,self.BPTT_length,self.num_inputs],name ='input_01')
		input_01=tf.reshape(self.input_01,[-1,self.BPTT_length,self.num_inputs,1])
		self.labels_1 = tf.placeholder(tf.int64, shape=[None], name = 'labels_1')
		self.labels_01 = tf.placeholder(tf.int64,shape=[None], name = 'labels_01')
        #self.keep_prob = tf.placeholder(tf.float32, name ='Drop_out_keep_prob')
		self.keep_prob_1 = tf.placeholder(tf.float32, name ='Drop_out_keep_prob_1')
		self.keep_prob_01 = tf.placeholder(tf.float32, name ='Drop_out_keep_prob_01')


		#first CNN layer
		h_1 = slim.conv2d(input_1, num_filters_per_size[0], [filter_size, self.num_inputs], normalizer_fn=slim.batch_norm, scope = 'conv0_1', padding='VALID')
		h_01 = slim.conv2d(input_01, num_filters_per_size[0], [filter_size, self.num_inputs], normalizer_fn=slim.batch_norm, scope = 'conv0_01', padding='VALID')
		# ================ Conv Block 64, 128, 256, 512 ================

		for i in range(0,len(num_filters_per_size)):
			for j in range(0,num_rep_block[i]):
				h_1 = resUnit(h_1, num_filters_per_size[i], filter_size, i, j)
			h_1 = slim.max_pool2d(h_1, [pooling_filter_size, 1], scope='pool_1_%s' % i)
		# ================ Layer FC ================

		# ================ Conv Block 64, 128, 256, 512 ================

		for i in range(0,len(num_filters_per_size)):
			for j in range(0,num_rep_block[i]):
				h_01 = resUnit(h_01, num_filters_per_size[i], filter_size, i, j)
			h_01 = slim.max_pool2d(h_01, [pooling_filter_size, 1], scope='pool_01_%s' % i)
		# ================ Layer FC ================
		# Global avg max pooling
		h_1 = math_ops.reduce_mean(h_1, [2, 1], name='pool5_1', keep_dims=True)
		h_01 = math_ops.reduce_mean(h_01, [2, 1], name='pool5_01', keep_dims=True)
		# Conv
		h_1 = slim.conv2d(h_1, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='output_1')
		h_01 = slim.conv2d(h_01, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='output_01')
		# FC & Dropout
		scores_1 = slim.flatten(h_1)
		scores_01 = slim.flatten(h_01)

		#fully connected layer
		output = tf.concat(1, [scores_1, scores_01])
		dense = tf.layers.dense(inputs=output, units = 10, activation = tf.nn.relu)
		scores = tf.layers.dense(inputs=dense, units=3)

		self.u_prob=tf.nn.softmax(scores)
		pred1D = tf.argmax(scores, 1, name="predictions")

		# ================ Loss and Accuracy ================
		# CalculateMean cross-entropy loss
		with tf.name_scope("evaluate"):
			losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores,labels=self.labels_01)
			self.loss = tf.reduce_mean(losses)
			correct_predictions = tf.equal(pred1D, self.labels_01)
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
			
		with tf.name_scope("Optimizer"):
			optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = beta1, beta2 = beta2, epsilon = epsilonADAM)
			self.train = optimizer.minimize(self.loss)

		self.init_op = tf.global_variables_initializer()
		print('Finished computation graph')


def MultiCNN(stock_ticker, x_batch_1, y_batch_1, x_batch_01, y_batch_01):
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
	BPTT_length = 20
	batch_size = 1024
	cnn_filter_size=3
	num_filters_per_size=(64,128,256,512)
	num_rep_block=(1,1,1,1) 
	pooling_filter_size=2
	LSTM_lasting_time = 20*BPTT_length
	train_dropout_1 = 0.8
	train_dropout_01 = 0.8
	test_dropout = 1.0
	learning_rate = 0.001
	beta1 = 0.9
	beta2 = 0.999
	epsilonADAM = 1e-8
	#*****************************************
	epoch_limit = 400

	config = {    'num_inputs' :    num_inputs,               
	              'filter_size' :   cnn_filter_size,         
	              'BPTT_length' :    BPTT_length,
	              'batch_size' :    batch_size,
	              'num_filters_per_size': num_filters_per_size,
	              'num_rep_block': num_rep_block, 
	              'beta1'  :        beta1,  
	              'beta2'  :        beta2,  
	              'pooling_filter_size': pooling_filter_size,         
	              'learning_rate' : learning_rate,
	              'epsilonADAM' : epsilonADAM,          
	              'num_classes':    num_classes}


	HP = '_' + 'num_rep_block' + str(num_rep_block) + '_' + 'batches'+ str(batch_size) + '_' + 'BPTT_length' + str(BPTT_length) +'_' + 'Dropout' + str(train_dropout_1)

	model = MultiCNN_Model(config)

	sess = tf.Session()
	sess.run(model.init_op)
	saver = tf.train.Saver()


	#save path
	header = '/home/leifan/Result/Comb_CNN_LSTM/'
	# os.system('mkdir'+ ' '+header+stock_name+'_01')
	os.system('mkdir'+ ' '+header+stock_ticker+'_Dual')
	path = header+stock_ticker+'_Dual/'
	os.system('mkdir'+ ' '+path+'save')
	save_path = path+'save/'
	path_saver = save_path + "Test" + stock_ticker + HP + ".ckpt"

	saver.restore(sess, path_saver)
    actual_out,probs = sess.run([model.labels_01, model.u_prob],
        feed_dict = {model.input_1: X_test_batch_1, model.labels_1: y_test_batch_1,
                    model.input_01: X_test_batch_01, model.labels_01: y_test_batch_01,
                    model.keep_prob_1:test_dropout, model.keep_prob_01:test_dropout})
    return actual_out, probs