import tensorflow as tf
import numpy as np
import pickle as pkl
import math

from utils.models_utils import *

OUTPUT_DIM = 6
INPUT_DIM = 400
SQRT_INPUT_DIM  =20 #IN ORDER TO RESHAPE INTO TENSOR
CONV_WINDOW_SIZE = 5
NUM_OF_CHANNELS_LAYER1 = 1
NUM_OF_CHANNELS_LAYER2 = 16
NUM_OF_CHANNELS_LAYER3 = 32
SIZE_OF_FULLY_CONNECTED_LAYER = 256
MAX_GAMES = 100
BATCH_SIZE = 5

VAR_NO = 8

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


observations = tf.placeholder(tf.float32, [INPUT_DIM]) #TODO: is tne None needed?
#trainable variables
weights = {'W_conv1': tf.Variable(tf.random_normal([CONV_WINDOW_SIZE , CONV_WINDOW_SIZE , NUM_OF_CHANNELS_LAYER1 , NUM_OF_CHANNELS_LAYER2])),
           'W_conv2': tf.Variable(tf.random_normal([CONV_WINDOW_SIZE , CONV_WINDOW_SIZE , NUM_OF_CHANNELS_LAYER2, NUM_OF_CHANNELS_LAYER3])),
           'W_fc': tf.Variable(tf.random_normal([CONV_WINDOW_SIZE  * CONV_WINDOW_SIZE  * NUM_OF_CHANNELS_LAYER3, SIZE_OF_FULLY_CONNECTED_LAYER])),
           'out': tf.Variable(tf.random_normal([SIZE_OF_FULLY_CONNECTED_LAYER, OUTPUT_DIM]))}

biases = {'b_conv1': tf.Variable(tf.random_normal([NUM_OF_CHANNELS_LAYER2])),
          'b_conv2': tf.Variable(tf.random_normal([NUM_OF_CHANNELS_LAYER3])),
          'b_fc': tf.Variable(tf.random_normal([SIZE_OF_FULLY_CONNECTED_LAYER])),
          'out': tf.Variable(tf.random_normal([OUTPUT_DIM]))}

#CNN:
x = tf.reshape(observations, shape=[-1, SQRT_INPUT_DIM, SQRT_INPUT_DIM, NUM_OF_CHANNELS_LAYER1])
#first layer: conv + pool
conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
conv1 = maxpool2d(conv1)
#second layer: conv + pool
conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
conv2 = maxpool2d(conv2)
#last layer - fully connected layer?
fc = tf.reshape(conv2, [-1, CONV_WINDOW_SIZE * CONV_WINDOW_SIZE * NUM_OF_CHANNELS_LAYER3])
fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
fc = tf.nn.dropout(fc, keep_rate)

score = tf.matmul(fc, weights['out']) + biases['out']
actions_probs = tf.nn.softmax(score)

tvars = tf.trainable_variables()
# rewards sums from k=t to T:
rewards_arr = tf.placeholder(tf.float32, [1,None])
# actions - a mask matrix which filters ys result accordint to the actions that were chosen.
actions_mask = tf.placeholder(tf.bool, [None, OUTPUT_DIM])
# return a T size vector with correct (chosen) action values
filtered_actions = tf.boolean_mask(actions_probs, actions_mask)
pi = tf.log(filtered_actions)
#devide by T
#loss = tf.divide(tf.reduce_sum(tf.multiply(pi,rewards_arr)),tf.to_float(tf.size(pi)))
#don't devide by T
loss = tf.reduce_sum(tf.multiply(pi,rewards_arr))
Gradients = tf.gradients(-loss,tvars)

Gradients_holder = [tf.placeholder(tf.float32) for i in range(VAR_NO)]
# then train the network - for each of the parameters do the GD as described in the HW.
#learning_rate = tf.placeholder(tf.float32, shape=[])   #TODO: maybe use later for oprimization of the model
train_step = tf.train.AdamOptimizer(1e-2).apply_gradients(zip(Gradients_holder,tvars))


#agent starts here:
init = tf.global_variables_initializer()
def main():
    rewards, states, actions_booleans = [], [], []
    episode_number = 0
    game_counter = 0

    #variables for debugging:
    manual_prob_use = 0

    with tf.Session() as sess:
        sess.run(init)

        #TODO: load wieghts


        update_weights = False #if to much time passed, update the weights even if the game is not finished
        grads_sums = get_empty_grads_sums()  # initialize the gradients holder for the trainable variables

        while game_counter < MAX_GAMES:
            obsrv, reward, done = get_observation()  # get observation
            # append the relevant observation to folloeing action, to states
            states.append(obsrv)        #TODO: not sure it is possible to agregate the state like this since this is not a matrix multipication
                                        #TODO: in the model but each time we send a tensor to the model... consider ask in ML-QA/stack overflow group
                                        #TODO: meanwhile create a model that is being updated after each observation
            # Run the policy network and get a distribution over actions
            action_probs = sess.run(actions_probs, feed_dict={observations: obsrv})
            # np.random.multinomial cause problems
            try:
                m_actions = np.random.multinomial(1, action_probs[0])
            except:
                m_actions = pick_random_action_manually(action_probs[0])
                manual_prob_use += 1

            # Saves the selected action for a later use
            actions_booleans.append(m_actions)
            # index of the selected action
            action = np.argmax(actions_booleans[-1])
            print("action chosen: " + str(action))
            # step the environment and get new measurements
            send_action(action)
            # add reward to rewards for a later use in the training step
            rewards.append(reward)
            game_counter += 1  #TODO: this is for tests

            #TODO: temporary, change to something that make sense...
            if(game_counter % 5 ==0):
                update_weights = True

            #TODO: sleep here?

            if done or update_weights:
                #UPDATE MODEL:
                '''
                # create the rewards sums of the reversed rewards array
                rewards_sums = np.cumsum(rewards[::-1])
                # normalize prizes and reverse
                rewards_sums = decrese_rewards(rewards_sums[::-1])
                rewards_sums -= np.mean(rewards_sums)
                rewards_sums = np.divide(rewards_sums, np.std(rewards_sums))
                modified_rewards_sums = np.reshape(rewards_sums, [1, len(rewards_sums)])
                # modify actions_booleans to be an array of booleans
                actions_booleans = np.array(actions_booleans)
                actions_booleans = actions_booleans == 1
                # gradients for current episode
                grads = sess.run(Gradients, feed_dict={observations: states, actions_mask: actions_booleans,
                                                       rewards_arr: modified_rewards_sums})
                grads_sums += np.array(grads)

                episode_number += 1
                update_weights = False

                # Do the training step
                if (episode_number % BATCH_SIZE == 0):
                    grad_dict = {Gradients_holder[i]: grads_sums[i] for i in range(VAR_NO)}
                    #TODO choose learning rate?
                    # take the train step
                    sess.run(train_step, feed_dict=grad_dict)
                    # nullify relevant vars and updates episode number.
                    rewards, states, actions_booleans = [], [], []
                    manual_prob_use = 0
                    grads_sums = get_empty_grads_sums()
            '''

main()




