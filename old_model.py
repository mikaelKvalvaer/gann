import tensorflow as tf
import main

# activation functions: tanh. just because (also try sigmoid)
# nodes in each hidden layer. This one is constant here..
# learning rate: 0.01
# W range: -0.08 to +0.08 (arbitrary)
# hidden layers: let's do 3!
# optimizer: we'll just do gradient descent
# output activation: softmax (one hot)
# loss function: cross-entropy

# network parameters
n_epochs = 15
batch_size = 500
display_step = 1  # ?

learning_rate = 0.25
n_hidden_1 = 30
n_hidden_2 = 25
n_hidden_3 = 20
n_input = 28 * 28
n_classes = 10

X = tf.placeholder(tf.float32, [batch_size, n_input], name='X')
Y = tf.placeholder(tf.int32, [batch_size], name='Y')
#    W = tf.get_variable('W', [FLAGS.num_rnn_cells, FLAGS.vocab_size], tf.float32,
    #                        tf.contrib.layers.xavier_initializer(), trainable=True)
    #
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.1)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=0.1)),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes], stddev=0.1))
}

biases = {
    'h1': tf.Variable(tf.random_normal([n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def multilayer_perceptron(x):

    # x = [100, 784], weights.shape = [784, 256] --> [100, 256]

    h1 = (tf.matmul(x, weights['h1']) + biases['h1'])
    h2 = (tf.matmul(h1, weights['h2']) + biases['h2'])
    h3 = tf.matmul(h2, weights['h3']) + biases['h3']
    out = (tf.matmul(h3, weights['out']) + biases['out'])
    # 83% prediction with h1 and h2

    # print('h2.hape = ', h2.shape)
    return out
    # return tf.tanh(h1)
    # return tf.tanh(tf.matmul(h1, weights['out']) + biases['out'])


logits = multilayer_perceptron(X)
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                      logits=logits, labels=Y))

print('logits shape is', logits.shape)
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)
init = tf.global_variables_initializer()

images, labels = main.load_dataset()
g = main.get_batches(images, labels, batch_size)
a,b = g.__next__()


with tf.Session() as s:
    s.run(init)

    # g = main.get_batches(images, labels, batch_size)
    for batch_no, (img, labels) in enumerate(g):
        feed_dict = {
            X: img,
            Y: labels
        }
        loss, _ = s.run([loss_op, train_op], feed_dict)
        print(f'step no {batch_no} with loss {loss}')

    predictions = tf.nn.softmax(logits)
    score = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1, output_type=tf.int32), Y), tf.float32))
    print(f'score is {score.eval({X: a, Y: b})}')
    vals = predictions.eval({X:a, Y:b})
    # print(f'predictions: {vals}')

