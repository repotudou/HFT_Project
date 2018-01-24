import tensorflow as tf
# Create some variables.
with tf.variable_scope("create2"):
	v1 = tf.get_variable("v1", shape=[3])
	v2 = tf.get_variable("v2", shape=[5])

	inc_v1 = v1.assign(v1+1)
	dec_v2 = v2.assign(v2-1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.

with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  inc_v1.op.run()
  dec_v2.op.run()
  # Save the variables to disk.
  save_path = saver.save(sess, "/Users/leifan/Dropbox/Comb/test/model2/model.ckpt")
  print("Model saved in file: %s" % save_path)