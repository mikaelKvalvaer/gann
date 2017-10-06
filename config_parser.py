import tensorflow as tf
import re


def parse_config(filename):
    with open(filename) as f:
        L = list(f)

    D = {}
    for line in L:
        if ':' not in line:  # line is useless, skip
            continue

        key, value = line.strip().strip(' ').split(':')
        if value.endswith(','):
            value = value[:-1]
        value = value.strip(' ')
        print(key, value)

        is_array = re.match('\[.*\]', value) is not None

        if is_array:
            the_array = eval(re.match('\[.*\]', value).group())
            D[key] = the_array
            continue

        is_numeric = re.match('^\d+\.?\d*$', value) is not None
        if is_numeric:
            if '.' in value:
                D[key] = float(value)
            else:
                D[key] = int(value)
            continue

        # else it's a string
        D[key] = value

    ##############################################
    # determine correct activation function
    ##############################################

    h_fun = resolve_activation_function(
        D['hidden_activation_function'])
    o_fun = resolve_activation_function(
        D['output_activation_function'])

    D['hidden_activation_function'] = h_fun
    D['output_activation_function'] = o_fun

    return D


def resolve_activation_function(func_name):
    return {'softmax': tf.nn.softmax, 'sigmoid': tf.sigmoid, 'identity': tf.identity,
            'tanh': tf.tanh, 'relu': tf.nn.relu}[func_name]


def config_is_valid(config_dict):
    required_keys = '''batch_size case_fraction data_source display_biases
    display_weights hidden_activation_function initial_wight_range
    learning_rate loss_function ma-batch_size map_dendrograms map_layers
    network_dimensions outpt_activation_function stpes test_fraction
    validation_fraction'''.strip().split()

    return all([k in required_keys for k in config_dict.keys()])
