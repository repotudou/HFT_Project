import tensorflow as tf

filename = 'model.ckpt'
sess = tf.Session()
new_saver = tf.train.import_meta_graph('model.ckpt.meta')
new_saver.restore(sess, filename)
all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
print(all_vars)
# for v in all_vars:
#     v_ = sess.run(v)
#     print(v_)