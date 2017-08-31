from utils.models_utils import *
from utils.reward_utils import calc_reward_from_raw
from utils.log_utils import *
from utils.plot_utils import plot_graph
import numpy as np
import pickle as pkl
import os
import tensorflow as tf
#TODO: I am not sure if there is a problem here with the Qt thing or not

#CNN constants
OUTPUT_DIM = 32
SQRT_INPUT_DIM  = 20 #IN ORDER TO RESHAPE INTO TENSOR
INPUT_DIM = SQRT_INPUT_DIM ** 2
PLN = 2                     #Pool Layers Number
CONV_WINDOW_SIZE = int(SQRT_INPUT_DIM / 2**PLN)
NUM_OF_CHANNELS_LAYER1 = 1
NUM_OF_CHANNELS_LAYER2 = 16     #TODO: Is that really what suppose to be here?
NUM_OF_CHANNELS_LAYER3 = 32
SIZE_OF_FULLY_CONNECTED_LAYER_1 = 256
SIZE_OF_FULLY_CONNECTED_LAYER_2 = 128
SIZE_OF_FULLY_CONNECTED_LAYER_3 = 64
VAR_NO = 12      #number of Ws and bs (the variables)
KEEP_RATE = 0.95
keep_prob = tf.placeholder(tf.float32)      #TODO: do we use that?
UNIFORM_DIST = [[1.0/OUTPUT_DIM] * OUTPUT_DIM]
EPSILON_FOR_EXPLORATION = 0.05
LEARNING_RATE = 1e-4

#Model constants
EPISODE_SIZE = 50
BATCH_SIZE = 50
MAX_EPISODES = EPISODE_SIZE*BATCH_SIZE*10000

#STEPS_UNTIL_BACKPROP = 1000
WRITE_TO_LOG = 100
#Load and save constants
WEIGHTS_FILE = 'weights_tom.pkl'
BEST_WEIGHTS = 'best_weights_tom.pkl'
LOAD_WEIGHTS = False

#other constants:
BEGINING_SCORE = 10

#initialize logger:
logger_scores = Logger('Tom_scores')
logger_parameters = Logger('Tom_parmameters')



def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def InitializeVarXavier(var_name,var_shape):
    return tf.get_variable(name=var_name, shape=var_shape, dtype= tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())

initialize = InitializeVarXavier


observations = tf.placeholder(tf.float32, [None,INPUT_DIM])
#trainable variables
weights = {'W_conv1': initialize('wc1',[CONV_WINDOW_SIZE , CONV_WINDOW_SIZE , NUM_OF_CHANNELS_LAYER1 , NUM_OF_CHANNELS_LAYER2]),
           'W_conv2':  initialize('wc2',[CONV_WINDOW_SIZE , CONV_WINDOW_SIZE , NUM_OF_CHANNELS_LAYER2, NUM_OF_CHANNELS_LAYER3]),
           'W_fc1': initialize('wfc1',[CONV_WINDOW_SIZE  * CONV_WINDOW_SIZE  * NUM_OF_CHANNELS_LAYER3, SIZE_OF_FULLY_CONNECTED_LAYER_1]),
           'W_fc2': initialize('wfc2',[SIZE_OF_FULLY_CONNECTED_LAYER_1,SIZE_OF_FULLY_CONNECTED_LAYER_2 ]),
           'W_fc3': initialize('wfc3',[SIZE_OF_FULLY_CONNECTED_LAYER_2,SIZE_OF_FULLY_CONNECTED_LAYER_3 ]),
           'out': initialize('wo',[SIZE_OF_FULLY_CONNECTED_LAYER_3, OUTPUT_DIM])}


biases = {'b_conv1': initialize('bc1', [NUM_OF_CHANNELS_LAYER2]),
          'b_conv2': initialize('bc2', [NUM_OF_CHANNELS_LAYER3]),
          'b_fc1': initialize('bfc1', [SIZE_OF_FULLY_CONNECTED_LAYER_1]),
          'b_fc2': initialize('bfc2', [SIZE_OF_FULLY_CONNECTED_LAYER_2]),
          'b_fc3': initialize('bfc3', [SIZE_OF_FULLY_CONNECTED_LAYER_3]),
          'out': initialize('bo', [OUTPUT_DIM])}

#CNN:
x = tf.reshape(observations, shape=[-1, SQRT_INPUT_DIM, SQRT_INPUT_DIM, NUM_OF_CHANNELS_LAYER1])
#first layer: conv + pool
conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
pool1 = maxpool2d(conv1)
#second layer: conv + pool
conv2 = tf.nn.relu(conv2d(pool1, weights['W_conv2']) + biases['b_conv2'])
pool2 = maxpool2d(conv2)
#last layer - fully connected layer?
r_layer2 = tf.reshape(pool2, [-1, CONV_WINDOW_SIZE * CONV_WINDOW_SIZE * NUM_OF_CHANNELS_LAYER3])
#stacking 3 fully connected layers
fc1 = tf.nn.relu(tf.matmul(r_layer2, weights['W_fc1']) + biases['b_fc1'])
dropped_fc1 = tf.nn.dropout(fc1, KEEP_RATE)

fc2 = tf.nn.tanh(tf.matmul(dropped_fc1, weights['W_fc2']) + biases['b_fc2'])
dropped_fc2 = tf.nn.dropout(fc2, KEEP_RATE)

fc3 = tf.nn.tanh(tf.matmul(dropped_fc2, weights['W_fc3']) + biases['b_fc3'])
dropped_fc3 = tf.nn.dropout(fc3, KEEP_RATE)

score = tf.matmul(dropped_fc3, weights['out']) + biases['out']
actions_probs = tf.nn.softmax(score)

#HERE STARTS THE GRADIENT COMPUTATION
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

#from here starts update weights
Gradients_holder = [tf.placeholder(tf.float32) for i in range(VAR_NO)]
    # then train the network - for each of the parameters do the GD as described in the HW.
#learning_rate = tf.placeholder(tf.float32, shape=[])   #TODO: maybe use later for oprimization of the model
train_step = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(Gradients_holder, tvars))


#agent starts here:
init = tf.global_variables_initializer()
#init2 = tf.initialize_all_variables()
def main():
    #variables used for models logics
    raw_scores, states, actions_booleans = [BEGINING_SCORE], [], []
    episode_number = 0
    grads_sums = get_empty_grads_sums()  # initialize the gradients holder for the trainable variables
    step_counter = 0

    #variables for debugging:
    manual_prob_counter = 0         #TODO: consider use the diffrences from 1
    prob_deviation_sum = 0
    default_data_counter = 0  # counts number of exceptions in reading the observations' file (and getting a default data)


    #variables for evaluation:
    best_avg_batch_score = 0
    best_score_in_a_single_game = 0
    game_counter = 0
    episode_scores = []
    avg_scores_per_batch = []
    final_scores = []

    with tf.Session() as sess:
        sess.run(init)
        #sess.run(init2)     #TODO: check if this necessary

        # check if file is not empty
        if (os.path.isfile(WEIGHTS_FILE) and LOAD_WEIGHTS):
            # Load with shmickle
            with open(WEIGHTS_FILE, 'rb') as f:  # BEST_WEIGHTS
                for var, val in zip(tvars, pkl.load(f)):
                    sess.run(tf.assign(var, val))
            print("loaded weights successfully!")

        # creates file if it doesn't exisits:
        if (not os.path.isfile(WEIGHTS_FILE)):
            open(WEIGHTS_FILE, 'a').close()
        if (not os.path.isfile(BEST_WEIGHTS)):
            open(BEST_WEIGHTS, 'a').close()
            print("created weights file sucessfully!")


        while episode_number < MAX_EPISODES:
            # get observation
            obsrv, score, is_dead, request_id, default_obsrv, AI_action, AI_accel = get_observation()

            # TODO: for debug
            default_data_counter += default_obsrv

            #get the initial score per episode
            if step_counter == 0:
                initial_episode_score = score

            raw_scores.append(score)
            states.append(obsrv)

            # Run the policy network and get a distribution over actions - Exploitation
            action_probs = sess.run(actions_probs, feed_dict={observations: [obsrv]})

            #CHOOSE ACTION FROM THE GIVEN DISTRIBUTION

            # exploration
            if(np.random.binomial(1,EPSILON_FOR_EXPLORATION , 1)[0]):
                action_probs = UNIFORM_DIST
            try:
                chosen_actions = np.random.multinomial(1, action_probs[0])
            except:
                chosen_actions = pick_random_action_manually(action_probs[0])
                manual_prob_counter += 1
                prob_deviation_sum += np.abs(np.sum(action_probs) - 1)

            # Saves the selected action for a later use
            actions_booleans.append(chosen_actions)
            # index of the selected action
            action = np.argmax(actions_booleans[-1])

            # step the environment and get new measurements
            send_action(action, request_id)
            step_counter += 1

            # Just for logging
            if (episode_number % WRITE_TO_LOG == 0):
                logger_parameters.write_to_log("in episode number {} :".format(episode_number))
                logger_parameters.write_to_log("observation : {}".format(str(obsrv)))
                logger_parameters.write_to_log("action_probs : {}".format(str(action_probs)))
            #TODO: sleep here?
            time.sleep(0.05)

            if (is_dead or (step_counter % EPISODE_SIZE == 0)) and (len(raw_scores) >= 2):#todo: maybe >=

                #calculate the score for the current episode
                episode_scores.append(raw_scores[-1] - initial_episode_score)

                if is_dead:
                    game_counter += 1
                    final_scores.append(raw_scores[-1])
                    print('just died! with score : {}'.format(final_scores[-1]))
                    logger_scores.write_to_log("score for game number {} is: {}"
                                               .format(game_counter, raw_scores[-1]))
                    if best_score_in_a_single_game < final_scores[-1]:
                        logger_scores.write_to_log("new best score score ,in game number {}, is: {}"
                                                        .format(game_counter, final_scores[-1]))


                #calculate rewards from raw scores:
                processed_rewards = calc_reward_from_raw(raw_scores,is_dead)
                modified_rewards_sums = np.reshape(processed_rewards, [1, len(processed_rewards)])
                # modify actions_booleans to be an array of booleans
                actions_booleans = (np.array(actions_booleans)) == 1

                #TODO: showind process results for debugging:
                fa_res = sess.run(filtered_actions, feed_dict={observations: states, actions_mask: actions_booleans,
                                                       rewards_arr: modified_rewards_sums})
                pi_res = sess.run(pi, feed_dict={observations: states, actions_mask: actions_booleans,
                                                       rewards_arr: modified_rewards_sums})
                loss_res = sess.run(loss, feed_dict={observations: states, actions_mask: actions_booleans,
                                                       rewards_arr: modified_rewards_sums})

                # gradients for current episode
                grads = sess.run(Gradients, feed_dict={observations: states, actions_mask: actions_booleans,
                                                       rewards_arr: modified_rewards_sums})
                grads_sums += np.array(grads)

                episode_number += 1
                # nullify step_counter:
                step_counter = 0
                #print("done calculating grads")
                # Do the training step
                if (episode_number % BATCH_SIZE == 0):
                    print("taking the update step")
                    grad_dict = {Gradients_holder[i]: grads_sums[i] for i in range(VAR_NO)}
                    # take the train step
                    sess.run(train_step, feed_dict=grad_dict)
                    #write to logger
                    avg_scores_per_batch.append(np.average(episode_scores))
                    logger_scores.write_to_log("avarage score for batch {} is: {} ".format(episode_number / BATCH_SIZE ,
                                                                              avg_scores_per_batch[-1]))

                    # evaluate and save:
                    if (best_avg_batch_score < avg_scores_per_batch[-1]):
                        best_avg_batch_score = avg_scores_per_batch[-1]
                        # save with shmickle
                        with open(BEST_WEIGHTS, 'wb') as f:
                            pkl.dump(sess.run(tvars), f, protocol=2)
                        print('best avg score ,Saved best weights successfully!')
                        logger_scores.write_to_log(
                            "Current best avg score , in batch number {}, is {}  , ".format(episode_number / BATCH_SIZE ,
                                                                              avg_scores_per_batch[-1]))
                    # manual save
                    with open(WEIGHTS_FILE, 'wb') as f:
                        pkl.dump(sess.run(tvars), f, protocol=2)

                    # plot FINAL graphs
                    plot_graph(final_scores, "test - tom", "test - tom - scores.png")
                    plot_graph(avg_scores_per_batch, "test - tom", "test - tom - score by batches.png")

                    #prepare for next batch
                    logger_scores.write_spacer()
                    episode_scores = []
                    grads_sums = get_empty_grads_sums()


                # prepare for next episode.
                raw_scores, states, actions_booleans = [BEGINING_SCORE], [], []
                wait_for_game_to_start()

    ########################################end of TF session#####################################3

    #final logger update
    logger_scores.write_to_log("proportion of default data".format(default_data_counter/MAX_EPISODES))
    logger_scores.write_to_log("proportion of manual_prob".format(manual_prob_counter/MAX_EPISODES))
    logger_scores.write_to_log("avg size of prob_deviation from 1".format(prob_deviation_sum/manual_prob_counter))

    #plot FINAL graphs
    plot_graph(final_scores,"test - tom","test - tom - scores.png")
    plot_graph(avg_scores_per_batch,"test - tom","test - tom - score by batches.png")




main()