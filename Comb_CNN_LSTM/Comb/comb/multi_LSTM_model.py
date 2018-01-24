import numpy as np
import tensorflow as tf 
import time
import os
import sys
import random

from tensorflow.contrib.rnn import LSTMCell





class Multi_LSTM_Model():
    def __init__(self,config):
        num_layers = config['num_layers']
        hidden_size = config['hidden_size']
        self.BPTT_length = config['BPTT_length']
        # max_grad_norm = config['max_grad_norm']
        self.batch_size = config['batch_size']
        self.num_inputs = config['num_inputs']
        learning_rate = config['learning_rate']
        beta1 = config['beta1']
        beta2 = config['beta2']
        epsilonADAM = config['epsilonADAM']
        num_classes = config['num_classes']
        time_interval_ratio = config['time_interval_ratio']

        self.input_1 = tf.placeholder(tf.float32,[None,self.BPTT_length,self.num_inputs],name ='input_1')
        self.input_01 = tf.placeholder(tf.float32,[None,self.BPTT_length,self.num_inputs],name ='input_01')
        self.labels_1 = tf.placeholder(tf.int64,[None,self.BPTT_length],name = 'labels_1')
        self.labels_01 = tf.placeholder(tf.int64,[None,self.BPTT_length],name = 'labels_01')
        #self.keep_prob = tf.placeholder(tf.float32, name ='Drop_out_keep_prob')
        self.keep_prob_1 = tf.placeholder(tf.float32, name ='Drop_out_keep_prob_1')
        self.keep_prob_01 = tf.placeholder(tf.float32, name ='Drop_out_keep_prob_01')

        with tf.name_scope("LSTM_setup") as scope:
            def single_cell_1():
                with tf.variable_scope('lstm_1'):
                    return tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_size,state_is_tuple=True),output_keep_prob=self.keep_prob_1)
            def single_cell_01():
                with tf.variable_scope('lstm_01'):
                    return tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_size,state_is_tuple=True),output_keep_prob=self.keep_prob_01) 
            cell_1 = tf.contrib.rnn.MultiRNNCell([single_cell_1() for _ in range(num_layers)],state_is_tuple=True)
            cell_01 = tf.contrib.rnn.MultiRNNCell([single_cell_01() for _ in range(num_layers)],state_is_tuple=True)

        self.reset_state_1 = cell_1.zero_state(self.batch_size, tf.float32)
        self.reset_state_01 = cell_01.zero_state(self.batch_size, tf.float32)

        self.state_1 = tf.placeholder(tf.float32, [num_layers,2,self.batch_size,hidden_size], "state_1")
        self.state_01 = tf.placeholder(tf.float32, [num_layers,2,self.batch_size,hidden_size], "state_01")
        spu = tf.unstack(self.state_1, axis=0)
        rnn_tuple_state_1 = tuple( [tf.contrib.rnn.LSTMStateTuple(spu[idx][0], spu[idx][1]) for idx in range(num_layers)])
        spu = tf.unstack(self.state_01, axis=0)
        rnn_tuple_state_01 = tuple( [tf.contrib.rnn.LSTMStateTuple(spu[idx][0], spu[idx][1]) for idx in range(num_layers)])


        self.outputs_1,self.next_state_1 = tf.nn.dynamic_rnn(cell=cell_1,inputs=self.input_1, dtype=tf.float32,
                                                parallel_iterations=self.batch_size, initial_state = rnn_tuple_state_1,scope ='lstm_1') 
        self.outputs_01,self.next_state_01  =  tf.nn.dynamic_rnn(cell=cell_01,inputs= self.input_01,  dtype=tf.float32,
                                                parallel_iterations =self.batch_size ,initial_state = rnn_tuple_state_01,scope ='lstm_01')
        outputs_unstacked_1 =  tf.unstack(self.outputs_1, axis=1)
        outputs_unstacked_01 =  tf.unstack(self.outputs_01, axis=1)
        tmp_label_1  = tf.unstack(self.labels_1, axis=1)
        tmp_label_01  = tf.unstack(self.labels_01, axis=1)
        output_1 = outputs_unstacked_1[-1]
        output_01 = outputs_unstacked_01[-1]
        output = tf.concat(1, [output_1, output_01])
        label_1 = tmp_label_1[-1]
        label_01 = tmp_label_01[-1]
        label = label_01

    

        with tf.name_scope('Fully_connected') as scope:
            with tf.variable_scope("hidden_params"):
                hidden_w_1 = tf.get_variable("hidden_w_1",[hidden_size*2,hidden_size])
                hidden_b_1 = tf.get_variable("hidden_b_1",[hidden_size])
            hidden = tf.nn.xw_plus_b(output, hidden_w_1, hidden_b_1)
            with tf.variable_scope("softmax_params"):
                softmax_w = tf.get_variable("softmax_w",[hidden_size,num_classes])
                softmax_b = tf.get_variable("softmax_b",[num_classes])
            self.logits = tf.nn.xw_plus_b(hidden, softmax_w, softmax_b)

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=label,name = 'softmax')
            self.cost = tf.reduce_sum(loss) / self.batch_size

        with tf.name_scope("Evaluating_accuracy") as scope:
            correct_prediction = tf.equal(tf.argmax(self.logits,1),label)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        with tf.name_scope("Optimizer") as scope:
            optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = beta1, beta2 = beta2, epsilon = epsilonADAM)
            #, decay=0.9, momentum=0.0, epsilon=1e-10)
            self.train_op = optimizer.minimize(self.cost)

        self.init_op = tf.global_variables_initializer()

        print('Finished computation graph')


    


