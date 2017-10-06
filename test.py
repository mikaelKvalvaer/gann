import tensorflow as tf

# variable scope


with tf.variable_scope('v', reuse=None):
    x = tf.get_variable('bob', [2,3])
    x = tf.get_variable('bob', [2,3])
    x = tf.get_variable('bob', [2,3])

with tf.variable_scope('v', reuse=True):
    y = tf.get_variable('bob')

print(x.name)
# y = tf.get_variable('bob')
