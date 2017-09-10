import tensorflow as tf
import numpy as np
import math
import json

with open('parameters/CNN_Params.json') as json_data:
    CNN_params = json.load(json_data)

#CNN constants
NUM_OF_FRAMES = CNN_params['FRAMES_PER_OBSERVATION']
OUTPUT_DIM = CNN_params['OUTPUT_DIM']
INPUT_DIM = CNN_params['INPUT_DIM']
SQRT_INPUT_DIM  = int(INPUT_DIM**0.5)
PLN = CNN_params['PLN']                     #Pool Layers Number
CONV_WINDOW_SIZE = int(SQRT_INPUT_DIM / 2**PLN)
NUM_OF_CHANNELS_LAYER1 = CNN_params['NUM_OF_CHANNELS_LAYER1']
NUM_OF_CHANNELS_LAYER2 = CNN_params['NUM_OF_CHANNELS_LAYER2']
NUM_OF_CHANNELS_LAYER3 = CNN_params['NUM_OF_CHANNELS_LAYER3']
SIZE_OF_FULLY_CONNECTED_LAYER_1 = CNN_params['SIZE_OF_FULLY_CONNECTED_LAYER_1']
SIZE_OF_FULLY_CONNECTED_LAYER_2 = CNN_params['SIZE_OF_FULLY_CONNECTED_LAYER_2']
SIZE_OF_FULLY_CONNECTED_LAYER_3 = CNN_params['SIZE_OF_FULLY_CONNECTED_LAYER_3']
SIZE_OF_FINAL_FC = SIZE_OF_FULLY_CONNECTED_LAYER_1

NUMBER_OF_FC_LAYERS = CNN_params['NUMBER_OF_FC_LAYERS']

KEEP_RATE = CNN_params['KEEP_RATE']

#checking the size of the final fc:
if(NUMBER_OF_FC_LAYERS == 2):
    print("the net will have 2 FC layers")
    SIZE_OF_FINAL_FC = SIZE_OF_FULLY_CONNECTED_LAYER_2
elif(NUMBER_OF_FC_LAYERS >= 3):
    print("the net will have 3 FC layers")
    SIZE_OF_FINAL_FC = SIZE_OF_FULLY_CONNECTED_LAYER_3



def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


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
    reshape_input = tf.reshape(input_layer, shape=[-1, SQRT_INPUT_DIM, SQRT_INPUT_DIM, NUM_OF_FRAMES])
    #first layer: conv + pool
    conv1 = create_conv_layer(reshape_input,'wc1',[CONV_WINDOW_SIZE, CONV_WINDOW_SIZE , NUM_OF_FRAMES, NUM_OF_CHANNELS_LAYER2],
                              'bc1', [NUM_OF_CHANNELS_LAYER2], with_polling=True)
    #second layer: conv + pool
    conv2 = create_conv_layer(conv1,'wc2',[CONV_WINDOW_SIZE , CONV_WINDOW_SIZE , NUM_OF_CHANNELS_LAYER2, NUM_OF_CHANNELS_LAYER3],
                              'bc2',[NUM_OF_CHANNELS_LAYER3], with_polling=True)
    #flatten layer
    r_layer2 = tf.reshape(conv2, [-1, CONV_WINDOW_SIZE * CONV_WINDOW_SIZE * NUM_OF_CHANNELS_LAYER3])
    #fully connected 1
    fc = create_fully_connected_layer(r_layer2, tf.nn.relu, 'wfc1',[CONV_WINDOW_SIZE  * CONV_WINDOW_SIZE  * NUM_OF_CHANNELS_LAYER3, SIZE_OF_FULLY_CONNECTED_LAYER_1],
                                       'bfc1',[SIZE_OF_FULLY_CONNECTED_LAYER_1], with_dropout=True)
    #fully connected 2
    if(NUMBER_OF_FC_LAYERS >= 2):
        fc = create_fully_connected_layer(fc, tf.nn.tanh, 'wfc2',[SIZE_OF_FULLY_CONNECTED_LAYER_1, SIZE_OF_FULLY_CONNECTED_LAYER_2],
                                           'bfc2',[SIZE_OF_FULLY_CONNECTED_LAYER_2], with_dropout=True)
    #fully connected 3
    if (NUMBER_OF_FC_LAYERS >= 3):
        fc = create_fully_connected_layer(fc, tf.nn.tanh, 'wfc3',[SIZE_OF_FULLY_CONNECTED_LAYER_2, SIZE_OF_FULLY_CONNECTED_LAYER_3],
                                           'bfc3',[SIZE_OF_FULLY_CONNECTED_LAYER_3], with_dropout=True)
    #output layer
    w_out, b_out = create_weights_and_biases('wout',[SIZE_OF_FINAL_FC, OUTPUT_DIM], 'bout', [OUTPUT_DIM] )
    score = tf.matmul(fc, w_out) + b_out

    #in DQN we calculate the expected reward. In models where the policy is calculated, remove this code from remark.
    #actions_probs = tf.nn.softmax(score)

    return input_layer, score