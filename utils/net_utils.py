import tensorflow as tf
import numpy as np

#CNN constants
NUM_OF_FRAMES = 4
OUTPUT_DIM = 64
SQRT_INPUT_DIM  = 20 #IN ORDER TO RESHAPE INTO TENSOR
INPUT_DIM = SQRT_INPUT_DIM*SQRT_INPUT_DIM
PLN = 2                     #Pool Layers Number
CONV_WINDOW_SIZE = int(SQRT_INPUT_DIM / 2**PLN)
NUM_OF_CHANNELS_LAYER2 = 16     #TODO: Is that really what suppose to be here?
NUM_OF_CHANNELS_LAYER3 = 32
SIZE_OF_FULLY_CONNECTED_LAYER_1 = 256
SIZE_OF_FULLY_CONNECTED_LAYER_2 = 128
SIZE_OF_FULLY_CONNECTED_LAYER_3 = 64

KEEP_RATE = 0.9



def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#TODO: is that initializer fine?
def InitializeVarXavier(var_name,var_shape):
    return tf.get_variable(name=var_name, shape=var_shape, dtype= tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())

initialize = InitializeVarXavier

def create_weights_and_biases(w_name,w_shape,b_name, b_shape):
    w = initialize(w_name,w_shape)
    b = initialize(b_name,b_shape)
    return w,b


def create_conv_layer(input, w_name,w_shape,b_name, b_shape, with_polling = True):
    w,b = create_weights_and_biases(w_name,w_shape,b_name, b_shape)
    res = tf.nn.relu(conv2d(input, w) + b)
    if(with_polling):
        res = maxpool2d(res)

    return res

def create_fully_connected_layer(input, f, w_name, w_shape, b_name, b_shape, with_dropout = True):
    w,b = create_weights_and_biases(w_name, w_shape, b_name, b_shape)
    res = f(tf.matmul(input, w) + b)
    if(with_dropout):
        res = tf.nn.dropout(res, KEEP_RATE)

    return res

def create_CNN():

    input_layer = tf.placeholder(tf.float32, [None,NUM_OF_FRAMES,INPUT_DIM])

    #CNN:
    reshape_input = tf.reshape(input_layer, shape=[-1, SQRT_INPUT_DIM, SQRT_INPUT_DIM,NUM_OF_FRAMES])
    #first layer: conv + pool
    conv1 = create_conv_layer(reshape_input,'wc1',[CONV_WINDOW_SIZE , CONV_WINDOW_SIZE ,  NUM_OF_FRAMES , NUM_OF_CHANNELS_LAYER2],
                              'bc1', [NUM_OF_CHANNELS_LAYER2], with_polling=True)
    #second layer: conv + pool
    conv2 = create_conv_layer(conv1,'wc2',[CONV_WINDOW_SIZE , CONV_WINDOW_SIZE , NUM_OF_CHANNELS_LAYER2, NUM_OF_CHANNELS_LAYER3],
                              'bc2',[NUM_OF_CHANNELS_LAYER3], with_polling=True)
    #flatten layer
    r_layer2 = tf.reshape(conv2, [-1, CONV_WINDOW_SIZE * CONV_WINDOW_SIZE * NUM_OF_CHANNELS_LAYER3])
    #fully connected 1
    fc1 = create_fully_connected_layer(r_layer2, tf.nn.relu, 'wfc1',[CONV_WINDOW_SIZE  * CONV_WINDOW_SIZE  * NUM_OF_CHANNELS_LAYER3, SIZE_OF_FULLY_CONNECTED_LAYER_1],
                                       'bfc1',[SIZE_OF_FULLY_CONNECTED_LAYER_1], with_dropout=True)
    #fully connected 2
    fc2 = create_fully_connected_layer(fc1, tf.nn.tanh, 'wfc2',[SIZE_OF_FULLY_CONNECTED_LAYER_1, SIZE_OF_FULLY_CONNECTED_LAYER_2],
                                       'bfc2',[SIZE_OF_FULLY_CONNECTED_LAYER_2], with_dropout=True)
    #fully connected 3
    fc3 = create_fully_connected_layer(fc2, tf.nn.tanh, 'wfc3',[SIZE_OF_FULLY_CONNECTED_LAYER_2, SIZE_OF_FULLY_CONNECTED_LAYER_3],
                                       'bfc3',[SIZE_OF_FULLY_CONNECTED_LAYER_3], with_dropout=True)
    #output layer
    w_out, b_out = create_weights_and_biases('wout',[SIZE_OF_FULLY_CONNECTED_LAYER_3, OUTPUT_DIM], 'bout', [OUTPUT_DIM] )
    score = tf.matmul(fc3, w_out) + b_out

    #TODO: in DQN we calculate the expected reward, and in the examples I saw they didn't softmax the result
    #actions_probs = tf.nn.softmax(score)

    return input_layer, score