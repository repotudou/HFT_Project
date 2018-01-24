import tensorflow as tf
import tensorflow.contrib.slim as slim
tf.reset_default_graph()

# Create some variables.
with tf.variable_scope("create1"):
	v1 = tf.get_variable("v1", shape=[3])
	# v2 = tf.get_variable("v2", shape=[5])
with tf.variable_scope("create2"):
	v3 = tf.get_variable("v1", shape=[3])
	# v4 = tf.get_variable("v2", shape=[5])


all_vars = tf.all_variables()
model_one_vars = [k for k in all_vars if k.name.startswith("create1")]
model_two_vars = [k for k in all_vars if k.name.startswith("create2")]

# if not model_two_vars:
#    self.saver_en_fr = tf.train.Saver(model_one_vars)
# else:
#    self.saver_fr_en = tf.train.Saver(model_two_vars)

# variables_to_restore = tf.get_collection(
#             slim.variables.VARIABLES_TO_RESTORE)
# print(model_one_vars)

# # Add ops to save and restore all the variables.
saver1 = tf.train.Saver(model_one_vars)
saver2 = tf.train.Saver(model_two_vars)
# # Later, launch the model, use the saver to restore variables from disk, and
# # do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver1.restore(sess, "/Users/leifan/Dropbox/Comb/test/model1/model.ckpt")
  saver2.restore(sess, "/Users/leifan/Dropbox/Comb/test/model2/model.ckpt")
  # saver.restore(sess, "/Users/leifan/Dropbox/Comb/test/model2/model.ckpt")
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  # print("v2 : %s" % v2.eval())
  print("v3 : %s" % v3.eval())
  # print("v4 : %s" % v4.eval())

# with tf.Session() as sess:    
#     saver = tf.train.import_meta_graph('/Users/leifan/Dropbox/Comb/test/model1/model.ckpt.meta')
#     saver.restore(sess,tf.train.latest_checkpoint('./'))
#     print(sess.run('v1:0'))
