import tensorflow as tf
import tflowtools as tt  # noqa
import utils as u
# show_results


class GANN():

    def __init__(self, config):
        self.config = config
        self.layers = []

    def build(self):
        tf.reset_default_graph()

        conf = self.config
        dims = conf['network_dimensions']
        input_dim = dims[0]

        x = tf.placeholder(tf.float32, [conf.batch_size, input_dim], name='x')  # noqa

        ##############################################
        # input and hidden layer generation
        ##############################################

        prev_activations = x
        for i, (dim1, dim2) in enumerate(zip(dims, dims[1:-1])):
            layer = Layer(i, prev_activations, dim1, dim2, self, conf)
            layer.build()
            prev_activations = layer.get_param('out')

        ##############################################
        # output layer generation
        ##############################################

        dim1, dim2 = dims[-2], dims[-1]

        i += 1  # increase the index also for the final layer.
        output_layer = Layer(i, prev_activations, dim1, dim2, self, conf, output_layer=True)
        output = output_layer.get_param('out')

        y = tf.pla

    def loss(predictions, target):
        # TODO: maybe return the loss value here..
        conf = self.config
        loss

        predictions
    def predict():
        pass
        # TODO: maybe give predictions here??

    def inference(self, inputs_x, inputs_y):
        conf = self.config
        dims = conf.network_dimensions
        last_hidden = len(dims) - 1
        lower, upper = conf.initial_weight_range
        input_dim = conf['network_dimensions'][0]

        x = tf.placeholder(tf.float32, [conf.batch_size, None], name='x')  # noqa
        y = tf.placeholder(tf.int32, [conf.batch_size], name='y')  # noqa

        print(f'network dimension {conf["network_dimensions"]}')

        with tf.variable_scope('hidden_layers'):
            for i, (s1, s2) in enumerate(zip([input_dim] + dims, dims)):
                tf.get_variable(f'weight_{i}', u.gen_normal([s1, s2]))
                tf.get_variable(f'bias_{i}', u.gen_normal([s2]))

        with tf.variable_scope('output_activation'):
            with tf.variable_scope('hidden_layers', reuse=True):
                h_last = tf.get_variable(f'weight_{last_hidden}')
                b_last = tf.get_variable(f'bias_{last_hidden}')

            arg = tf.matmul(x, h_last) + b_last
            logits = conf.output_activation_function(arg, name='logits')
            return logits

    def add_layer(self, layer):
        self.layers.append(layer)


    def loss(self, logits):
        conf = self.config
        graph = self.graph or tf.get_default_graph()

        # TODO: also open for other activation functions.
        if conf['output_activation_function'] == 'softmax':
            val = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits,
                    labels=graph.get_tensor_by_name('y:0'))

        loss = tf.reduce_mean(val)
        return loss

    def train(self, loss):
        pass


class Layer():

    def __init__(self, prev_activations, input_dim, output_dim,
                 ann, config, output_layer=False):

        self.input = prev_activations
        self.input_dim = input_dim
        self.outpu_dim = output_dim
        self.ann = ann
        self.config = config

        if output_layer:
            self.a_fun = self.config['output_activation_function']
        else:
            self.a_fun = self.config['hidden_activation_function']

    def build(self):
        '''
        Builds the layer that consists of weights and biases as well as
        the output value. Register itself to the network by calling
        the method `add_layer` of `ann` (the network it belongs to).
        '''

        self.weights = tf.Variable(
            u.gen_normal((self.input_dim, self.output_dim)),
            name=f'{self.name}_weight', trainable=True)

        self.biases = tf.Variable(
            u.gen_normal([self.output_dim]),
            name=f'{self.name}_bias', trainable=True)

        self.output = self.a_fun(
            tf.matmul(self.input, self.weights) + self.biases,
            name=f'{self.name}_out')

        self.ann.add_layer(self)

    def get_param(self, name):
        return {'in': self.input, 'out': self.output,
                'weight': self.weight, 'bias': self.biases}[name]

    def gen_probe(self, name, spec):
        var = self.get_param(name)
        base = f'{self.name}_{name}'

        with tf.name_scope('probe_'):
            if ('avg' in spec) or ('stdev' in spec):
                avg = tf.reduce_mean(var)
            if 'avg' in spec:
                tf.summary.scalar(base + '/avg/', avg)
            if 'max' in spec:
                tf.summary.scalar(base + '/max/', tf.reduce_max(var))
            if 'min' in spec:
                tf.summary.scalar(base + '/min/', tf.reduce_min(var))
            if 'hist' in spec:
                tf.summary.histogram(base + '/hist/',var)
