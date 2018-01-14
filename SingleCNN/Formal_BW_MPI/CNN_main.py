import numpy as np
import tensorflow as tf  
from CNN_model import load_train_test_data, CNN_Model
import random
from mpi4py import MPI
import os
import time

time1 = time.time()

node_name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
time_interval = 1

if time_interval==1:
    tailer = '_1'
elif time_interval==0.1:
    tailer = '_01'

stock_file_list = ['AMD', 'BIDU', 'RCL', 'CBS', 'D', 'SYMC', 'EMR', 'ENDP', 'UPS', 'HON', 'URBN','HRL', 'VFC', 'JNPR', 'VIAB', 'JWN', 'KMB']
#stock_file_list = ['AMD','BIDU']
stock_ticker = stock_file_list[rank]
# stock_ticker = 'AMD'

output_list = load_train_test_data(stock_ticker, time_interval)
if not output_list[0]:
    print("There is an error!")
    exit()

HDF5_file_LoadRNN = output_list[1]
X_np_train = HDF5_file_LoadRNN['X_train']
Y_np_train = HDF5_file_LoadRNN['y_train']
X_np_eval = HDF5_file_LoadRNN['X_test']
Y_np_eval = HDF5_file_LoadRNN['y_test']

print("Data Read",stock_ticker)

num_classes=3
num_levels = 10
num_inputs = num_levels*2+4
BPTT_length=80
batch_size = 500
cnn_filter_size=3
pooling_filter_size=2
num_filters_per_size=(64,128,256,512)
num_rep_block=(1,1,1,1) # how many layers in each block??
epoch_limit = 40
learning_rate = 0.001
epsilonADAM = 1e-8


T = 1000*batch_size
T_eval = 100*batch_size
summary_bool = False

keep_prob_train=0.95

config = {    'num_inputs' :    num_inputs,               
              'filter_size' :   cnn_filter_size,         
              'BPTT_length' :    BPTT_length,
              'batch_size' :    batch_size,
              'num_filters_per_size': num_filters_per_size,
              'num_rep_block': num_rep_block, 
              'beta1'  :        0.9,  
              'beta2'  :        0.999,  
              'pooling_filter_size': pooling_filter_size,         
              'learning_rate' : learning_rate,
              'epsilonADAM' : epsilonADAM,          
              'num_classes':    num_classes,
              'summary_bool': summary_bool}

CNN_model = CNN_Model(config)

#identify model
training_case = [learning_rate, num_rep_block, BPTT_length, pooling_filter_size, cnn_filter_size, num_filters_per_size, num_levels, epsilonADAM, keep_prob_train, T, T_eval, batch_size, stock_ticker]
HP = '_' + 'num_rep_block' + str(num_rep_block) + '_' + 'batches'+ str(batch_size) + '_' + 'time_lenght' + str(BPTT_length) +'_' + 'stock_file_name' + str(stock_ticker)

sess = tf.Session()
sess.run(CNN_model.init_op)

#save files
# header = '/u/sciteam/fan2/CNN_MPI2/'
header = '/u/sciteam/fan2/CNN_MPI2/'
os.system('mkdir'+ ' '+header+stock_ticker+tailer)
os.system('mkdir'+ ' '+header+stock_ticker+tailer+'/'+'save')
path = header+stock_ticker+tailer+'/'
save_path = path +'save/'
path_saver = save_path + "Test" + stock_ticker + HP + ".ckpt"

#save model
saver = tf.train.Saver()
#saver.restore(session, path_saver)


#write results in a file
file_name = stock_ticker + '_CNN.txt'

file_1 = open(file_name,'w') 
file_1.write('The following is for the stock %s in CNN Model with time interval %f  \n'%(stock_ticker,time_interval))

#define savable values
eval_error_tracker = []
train_error_tracker = []
pickle_list0 = [[], [], training_case]
train_err = np.zeros(epoch_limit)
test_err = np.zeros(epoch_limit)
train_acc = np.zeros(epoch_limit)
test_acc = np.zeros(epoch_limit)
cond_acc = np.zeros(epoch_limit)

for i in range(epoch_limit):
    print('epoch: %d' %(i+1))
    file_1.write('epoch: %d' %(i+1))
    current_train_error = 0.0
    current_train_acc = 0.0
    # TRAIN MODEL
    random_index_list = []
    for k in range(batch_size):
        random_index = random.randint(0,len(Y_np_train)-T-1)
        random_index_list.append(random_index)
    counter = 0
    train_error_avg=0
    x_batch = np.float32(np.zeros((batch_size, num_inputs,BPTT_length)))
    y_batch = np.int64(np.zeros((batch_size)))
    for t in range(0, T, batch_size):
        for k in range(batch_size):
            kk =random_index_list[k]
            x_batch[k, :, :] = np.transpose(X_np_train[t+kk :t+kk + BPTT_length, :])
            y_batch[k] = Y_np_train[t+kk + BPTT_length-1]
        _, train_accuracy, loss_np = sess.run([CNN_model.train, CNN_model.accuracy, CNN_model.loss],feed_dict={CNN_model.input_x1: x_batch, CNN_model.input_y: y_batch,CNN_model.dropout_keep_prob: keep_prob_train})
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
    sum1=0
    for k in range(batch_size):
        random_index = random.randint(0,len(Y_np_eval)-T_eval-1)
        random_index_list.append(random_index)
    counter = 0
    x_batch = np.float32(np.zeros((batch_size, num_inputs,BPTT_length)))
    y_batch = np.int32(np.zeros((batch_size)))
    Movements_predicted = 0.0
    for t in range(0, T_eval, batch_size):
        for k in range(batch_size):
            kk =random_index_list[k]
            x_batch[k, :, :] = np.transpose(X_np_eval[t+kk :t+kk + BPTT_length, :])
            y_batch[k] = Y_np_eval[t+kk + BPTT_length-1]
        loss_np, accuracy_np,actual_out,probs = sess.run([CNN_model.loss, CNN_model.accuracy,CNN_model.input_y,CNN_model.u_prob],feed_dict={CNN_model.input_x1: x_batch, CNN_model.input_y: y_batch,CNN_model.dropout_keep_prob: np.float32(1.0)})
        probs[:, 1] = 0
        pred_out = [np.argmax(probs, 1)]
        pred_out = np.array(pred_out).reshape(-1)
        current_eval_error += loss_np
        current_eval_acc += accuracy_np
        counter += 1
        Movements_predicted += sum([1 for p, j in zip(actual_out, pred_out) if p != 1 and (p == j)])
        total_movements += sum([1 for p, j in zip(actual_out, pred_out) if p!= 1])

    conditional_accuracy = np.array(Movements_predicted) / total_movements
    print(Movements_predicted, 'Movements_predicted correctly_testing')
    print(total_movements, 'Total Movements_testing')
    print(conditional_accuracy, 'Conditional Accuracy_testing')
    file_1.write("For the stock %s, Movements_predicted correctly_testing is %f\n" %(stock_ticker, Movements_predicted))
    file_1.write("For the stock %s, Total Movements_testing is %f\n" %(stock_ticker, total_movements))
    file_1.write("For the stock %s, Conditional Accuracy_testing is %f\n" %(stock_ticker, conditional_accuracy))
    # print(train_error_avg, 'training loss')
    accuracy_np_ave = current_eval_acc / np.float32(counter)
    loss_np_ave = current_eval_error / np.float32(counter)
    test_acc[i] = accuracy_np_ave
    test_err[i] = loss_np_ave
    cond_acc[i] = conditional_accuracy
    eval_error_tracker.append((train_loss_avg,accuracy_np_ave, loss_np_ave, conditional_accuracy))
    print(stock_ticker, 'train_err', train_err[i],  'test_err', test_err[i], 'train_acc', train_acc[i], 'test_acc', test_acc[i], 'conditional_acc', cond_acc[i])
    file_1.write('stock_ticker is %s, train_err is %f, test_err is %f, train_acc is %f, test_acc is %f, conditional_acc is %f\n' %(stock_ticker, train_err[i], test_err[i], train_acc[i], test_acc[i],cond_acc[i]))
    if (i+1)%5 == 0:
        saver.save(sess, path_saver)

time2 = time.time()
total_time = time2-time1
print('Total time =',time2-time1)
file_1.write('Total time = %s\n' %total_time)
file_1.close()


#save files
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
