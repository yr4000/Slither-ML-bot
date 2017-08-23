import numpy as np
import pickle
import tensorflow as tf
import pdb
import time
import random

BATCH_SIZE = 1
LR = 1e-2
REDUCE_BASE = 0.95
REDUCE_EPISODES = 3000
discount_factor = 0.99
OUTPUT_DIM = 8
INPUT_DIM = 400
SQRT_INPUT_DIM = 20 #IN ORDER TO RESHAPE INTO TENSOR
CONV_WINDOW_SIZE = 5
NUM_OF_CHANNELS_LAYER1 = 1
NUM_OF_CHANNELS_LAYER2 = 16
NUM_OF_CHANNELS_LAYER3 = 32
SIZE_OF_FULLY_CONNECTED_LAYER = 256
MAX_GAMES = 30

def discount_rewards(arr):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_arr = np.zeros_like(arr)
    cumulative = 0
    for t in reversed(range(0, arr.size)):
        cumulative = cumulative * discount_factor + arr[t]
        discounted_arr[t] = cumulative
    return discounted_arr


# TODO: what is dropout?!
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

# build the computational graph

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


#DEFINE TRAINABLE VARIABLES
weights = {'W_conv1': tf.Variable(tf.random_normal([CONV_WINDOW_SIZE , CONV_WINDOW_SIZE , NUM_OF_CHANNELS_LAYER1 , NUM_OF_CHANNELS_LAYER2])),
           'W_conv2': tf.Variable(tf.random_normal([CONV_WINDOW_SIZE , CONV_WINDOW_SIZE , NUM_OF_CHANNELS_LAYER2, NUM_OF_CHANNELS_LAYER3])),
           'W_fc': tf.Variable(tf.random_normal([CONV_WINDOW_SIZE  * CONV_WINDOW_SIZE  * NUM_OF_CHANNELS_LAYER3, SIZE_OF_FULLY_CONNECTED_LAYER])),
           'out': tf.Variable(tf.random_normal([SIZE_OF_FULLY_CONNECTED_LAYER, OUTPUT_DIM]))}

biases = {'b_conv1': tf.Variable(tf.random_normal([NUM_OF_CHANNELS_LAYER2])),
          'b_conv2': tf.Variable(tf.random_normal([NUM_OF_CHANNELS_LAYER3])),
          'b_fc': tf.Variable(tf.random_normal([SIZE_OF_FULLY_CONNECTED_LAYER])),
          'out': tf.Variable(tf.random_normal([OUTPUT_DIM]))}

#DEFINE feed forward computational graph
x = tf.reshape(x, shape=[-1, SQRT_INPUT_DIM, SQRT_INPUT_DIM, NUM_OF_CHANNELS_LAYER1])

conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
conv1 = maxpool2d(conv1)

conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
conv2 = maxpool2d(conv2)

fc = tf.reshape(conv2, [-1, CONV_WINDOW_SIZE * CONV_WINDOW_SIZE * NUM_OF_CHANNELS_LAYER3])
fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
fc = tf.nn.dropout(fc, keep_rate)

score = tf.matmul(fc, weights['out']) + biases['out']
probability = tf.nn.softmax(score)

observation = tf.placeholder(dtype=tf.float64) #TODO type

#DEFINE backpropogation computational graph
accu_reward = tf.placeholder(tf.float64,[None])
actual_acts = tf.placeholder(tf.int32,[None])
indices = tf.range(0, tf.shape(probability)[0]) * tf.shape(probability)[1] + actual_acts
probed_acts = tf.gather(tf.reshape(probability, [-1]), indices)
loglik = tf.log(probed_acts)
loss = -tf.reduce_sum(tf.multiply(loglik, accu_reward))
trainable = tf.trainable_variables()
newGrads = tf.gradients(loss, trainable)

gradient_holders = []
for idx,var in enumerate(trainable):
    placeholder = tf.placeholder(tf.float64, name=str(idx)+'_holder')
    gradient_holders.append(placeholder)

global_step = tf.Variable(0, trainable=False, name='glob_step')
exp_lr = tf.train.exponential_decay(LR, global_step, REDUCE_EPISODES, REDUCE_BASE, staircase=True)
adam = tf.train.AdamOptimizer(exp_lr)
updateGrads = adam.apply_gradients(zip(gradient_holders,trainable))

final_vars_to_save = tf.trainable_variables()

##########################################################
def run_model():
    begining_time = int(time.time())
    init = tf.global_variables_initializer()
    game_counter = 0
    max_avg_reward = 0
    reward_sum = 0
    saved_all_reward = []
    saved_all_actions = []
    all_observs = []
    all_actions = []
    all_rewards = []
    all_probs = []

    #running_reward = None
    #saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init)
        #cost = tf.reduce_mean(tf.nn.softmax(labels=y, logits=prediction))
        #optimizer = tf.train.AdamOptimizer().minimize(cost)
        prev_obsrv = env.reset() # Obtain an initial observation of the environment
        #gradBuffer = []
        
        gradBuffer = sess.run(tf.trainable_variables())
        for ix,grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0
        
        #first_act = random.randint(0,OUTPUT_DIM-1)
        #prev_obsrv, reward, done, info = env.step(first_act)
        
        while game_counter < MAX_GAMES:
            # Run the policy network and get a distribution over actions
            action_probs = sess.run(probability, feed_dict={observation: prev_obsrv.reshape(1,INPUT_DIM)})
            # sample action from distribution
            action = np.argmax(np.random.multinomial(1, action_probs.reshape(-1)))
            # step the environment and get new measurements
            obsrv, reward, done, info = env.step(action)
            
            all_observs.append(prev_obsrv)
            all_rewards.append(reward)
            all_actions.append(action)
            all_probs.append(action_probs.reshape(-1)[action])
            reward_sum += reward
            prev_obsrv = obsrv
            
            if done: 
                game_counter += 1
                
                discounted_rewards = discount_rewards(np.array(all_rewards))
                # size the rewards to be unit normal (helps control the gradient estimator variance)
                discounted_rewards -= np.mean(discounted_rewards)
                discounted_rewards //= np.std(discounted_rewards)
                #TODO: do we need to send all_observs?
                tGrad = sess.run(newGrads, feed_dict={observation: all_observs,\
                                                      actual_acts: all_actions,\
                                                     accu_reward: discounted_rewards})     
        
                for idx,grad in enumerate(tGrad):
                    gradBuffer[idx] += grad

                all_observs = []
                saved_all_reward.append(all_rewards)
                all_rewards = []
                saved_all_actions.append(all_actions)
                all_actions = []

                # every BATCH_SIZE episodes update the weights
                if game_counter % BATCH_SIZE == 0:
                    feed_dict = dict(zip(gradient_holders, gradBuffer))
                    feed_dict[global_step] = game_counter
                    curr_lr, _ = sess.run([exp_lr,updateGrads], feed_dict=feed_dict)
                    for ix,grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0

                    # Give a summary of how well our network is doing for each 10 games
                    if game_counter % 10 == 0:
                        print("Average reward for 10 episodes {0:.2f}\t eps: {1}\t time: {2:.2f}\t LR: {3:.5f}"\
                        .format(reward_sum/10.0, game_counter, time.time() - begining_time, curr_lr))
                        if reward_sum/10.0 > max_avg_reward:
                            cu_time = int(time.time())
                            max_avg_reward = reward_sum/10.0
                            print("New best average reward {0:.2f}, after {1} games, {2} seconds".format(reward_sum/10.0, game_counter, cu_time - begining_time))
                            #saver.save(sess, r'C:\Users\carmel\Desktop\courses\adv-ml\Ex04\models\{2}\{0}_{1}'.format(end_time, reward_sum/BATCH_SIZE, env_d))
                            with open(r'{0}\{1}'.format(MODEL_PATH, begining_time), 'wb') as f:
                                pdata = sess.run(final_vars_to_save,)
                                pickle.dump(pdata, f)
                            flat_list = [item for sublist in saved_all_actions for item in sublist]
                            uni, cou = np.unique(flat_list, return_counts=True)
                            print(dict(zip(uni,cou*1.0/len(flat_list))))
                            
                        
                        reward_sum = 0
                    
                #TODO: maybe send singnal to start a new game
                    
run_model()
