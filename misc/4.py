#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 23:11:02 2017

@author: daniel
"""
import gym
import numpy as np
import pickle as pkl
import tensorflow as tf
import matplotlib
matplotlib.use('Qt4Agg')
#matplotlib.use('Agg') #in order not to display the plot
import matplotlib.pyplot as plt
import os
import math


env_d = 'LunarLander-v2'
cart_pole_env = 'CartPole-v0'
INPUT_SIZE = 8 #for lunar it should be size 8, for cartPole 4
HIDDEN_NEURONS_NO = 15
OUTPUT_SIZE = 4 #for lunar it should be size 4, for cartPole 2
LAYERS_NO = 3
VAR_NO = LAYERS_NO*2
#LEARNING_RATE = 1e-2
TOTAL_EPISODES = 30000
PERIOD = 10
PLOT_PERIOD = 100
SAVE_PERIOD = 100
DO_NORMALIZE = True
LOAD = True
SAVE = True

ENVIRONMENT = env_d
#WEIGHTS_FILE = './'+ENVIRONMENT+'_weights.pkl'
if(DO_NORMALIZE):
    WEIGHTS_FILE = 'n-ws.p'
    BEST_WEIGHTS = 'n-bws.p'
else:
    WEIGHTS_FILE = 'ws.p'
    BEST_WEIGHTS = 'bws.p'

env = gym.make(ENVIRONMENT)
env.reset()


#define different initialize functions for variables
def InitializeVarXavier(var_name,var_shape):
    return tf.get_variable(name=var_name, shape=var_shape, dtype= tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())

def InitializeVarRandomNormal(var_name,var_shape):
    return tf.Variable(tf.random_normal(var_shape, stddev = 0.1), dtype=tf.float32)

#here we choose how to initialize
if(DO_NORMALIZE):
    print("Running at normalized mode.")
    initialize = InitializeVarRandomNormal
else:
    initialize = InitializeVarXavier


# Defining the agent
# placeholder for the input
observations = tf.placeholder(tf.float32, [None, INPUT_SIZE])
W1 = initialize("W1",[INPUT_SIZE, HIDDEN_NEURONS_NO])
b1 = initialize("b1",[HIDDEN_NEURONS_NO])
if (LAYERS_NO == 3):
    W2 = initialize("W2",[HIDDEN_NEURONS_NO, HIDDEN_NEURONS_NO])
    b2 = initialize("b2",[HIDDEN_NEURONS_NO])
W3 = initialize("W3",[HIDDEN_NEURONS_NO, OUTPUT_SIZE])
b3 = initialize("b3",[OUTPUT_SIZE])

#first,second and third layer computations
h1 = tf.nn.tanh(tf.matmul(observations, W1) + b1)
if (LAYERS_NO == 3):
    h2 = tf.nn.tanh(tf.matmul(h1, W2) + b2)
    y = tf.nn.softmax(tf.matmul(h2, W3) + b3)
else:
    y = tf.nn.softmax(tf.matmul(h1, W3) + b3)

tvars = tf.trainable_variables()
# rewards sums from k=t to T:
rewards_arr = tf.placeholder(tf.float32, [1,None])
# actions - a mask matrix which filters ys result accordint to the actions that were chosen.
actions_mask = tf.placeholder(tf.bool, [None, OUTPUT_SIZE])
# return a T size vector with correct (chosen) action values
filtered_actions = tf.boolean_mask(y, actions_mask)
pi = tf.log(filtered_actions)
#devide by T
#loss = tf.divide(tf.reduce_sum(tf.multiply(pi,rewards_arr)),tf.to_float(tf.size(pi)))
#don't devide by T
loss = tf.reduce_sum(tf.multiply(pi,rewards_arr))
Gradients = tf.gradients(-loss,tvars)

Gradients_holder = [tf.placeholder(tf.float32) for i in range(VAR_NO)]
# then train the network - for each of the parameters do the GD as described in the HW.
learning_rate = tf.placeholder(tf.float32, shape=[])
train_step = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(Gradients_holder,tvars))

#we assume here we get the array in the right order, so each sum is indeed being multiply with the right factor
def decrese_rewards(rewards):
    gama = 0.99
    dec_arr = np.array([gama**(len(rewards)-t) for t in range(len(rewards))])
    res = np.multiply(rewards,dec_arr)
    return res

#returns objects to hold the gradients of each trainable variable
def get_empty_grads_sums():
    grads_sums = tf.trainable_variables()
    for i, val in enumerate(grads_sums):
        grads_sums[i] = 0
    return grads_sums

def simple_normalize_vector(v):
    res = np.divide(v,np.sum(v))
    return res

def pick_random_action_manually(actions):
    r = 1
    while(r==1):
        r = np.random.uniform()
    m = np.max(actions)
    l = len(actions)
    if( r < m):
        res = [ 0 if actions[i] < m else 1 for i in range(l)]
    else:
        index = math.floor(r*l)
        res = [0 if i != index else 1 for i in range(l)]
    return res


init = tf.global_variables_initializer()
def main(argv):
    rewards, states, actions_booleans = [], [], []
    episode_number,period_reward,running_reward,reward_for_plot,save_reward,previous_period_reward = 0,0,0,0,None,None
    steps = 0
    manual_prob_use = 0             #for debug
    #for plotting:
    rewards_per_episode = [0 for i in range(TOTAL_EPISODES//PLOT_PERIOD)]


    with tf.Session() as sess:
        sess.run(init)
        #saver = tf.train.Saver()

        #check if file is not empty
        if(os.path.isfile(WEIGHTS_FILE) and LOAD):
            '''
            #Load with Tensorflow saver
            saver.restore(sess,WEIGHTS_FILE)
            '''
            #Load with shmickle
            f = open(BEST_WEIGHTS,'rb')     #BEST_WEIGHTS
            for var, val in zip(tvars,pkl.load(f)):
                sess.run(tf.assign(var,val))
            f.close()
            print("loaded weights successfully!")

        #creates file if it doesn't exisits:
        if(not os.path.isfile(WEIGHTS_FILE)):
            open(WEIGHTS_FILE,'a').close()
        if(not os.path.isfile(BEST_WEIGHTS)):
            open(BEST_WEIGHTS, 'a').close()
            print("created file sucessfully!")

        obsrv = env.reset() # Obtain an initial observation of the environment
        grads_sums = get_empty_grads_sums()     #initialize the gradients holder for the trainable variables

        while (episode_number <= TOTAL_EPISODES): #or (reward_for_plot/PLOT_PERIOD > 200):
            env.render()

            #append the relevant observation to folloeing action, to states
            states.append(obsrv)
            #modify the observation to the model
            modified_obsrv = np.reshape(obsrv, [1, INPUT_SIZE])
            # Run the policy network and get a distribution over actions
            action_probs = sess.run(y,feed_dict={observations: modified_obsrv})
            # np.random.multinomial cause problems
            try:
                m_actions = np.random.multinomial(1, action_probs[0])
            except:
                m_actions = pick_random_action_manually(action_probs[0])
                manual_prob_use += 1
            # Saves the selected action for a later use
            actions_booleans.append(m_actions)
            #index of the selected action
            action = np.argmax(actions_booleans[-1])
            # step the environment and get new measurements
            obsrv, reward, done, info = env.step(action)
            #add reward to rewards for a later use in the training step
            rewards.append(reward)
            steps += 1
            if done:
                #create the rewards sums of the reversed rewards array
                rewards_sums = np.cumsum(rewards[::-1])
                #normalize prizes and reverse
                rewards_sums = decrese_rewards(rewards_sums[::-1])
                rewards_sums -= np.mean(rewards_sums)
                rewards_sums = np.divide(rewards_sums, np.std(rewards_sums))
                modified_rewards_sums = np.reshape(rewards_sums, [1, len(rewards_sums)])
                #modify actions_booleans to be an array of booleans
                actions_booleans = np.array(actions_booleans)
                actions_booleans = actions_booleans == 1
                #gradients for current episode
                grads = sess.run(Gradients, feed_dict={observations: states,actions_mask:actions_booleans,rewards_arr: modified_rewards_sums })
                grads_sums += np.array(grads)

                #reward counters for printing and plotting.
                period_reward += sum(rewards)
                reward_for_plot += sum(rewards)

                episode_number += 1
                #saves the best weights:
                if(episode_number%PLOT_PERIOD==0):
                    if((save_reward is None or save_reward<(reward_for_plot/ PLOT_PERIOD)) and SAVE):
                        save_reward = (reward_for_plot/ PLOT_PERIOD)
                        '''
                        #save with Tensorflow saver:
                        #saver.save(sess,WEIGHTS_FILE)
                        '''
                        #save with shmickle
                        f = open(BEST_WEIGHTS,'wb')
                        pkl.dump(sess.run(tvars),f,protocol=2)
                        f.close()
                        print('Saved best weights successfully!')
                        print('Current best result for %d episodes: %f.' %(PLOT_PERIOD,reward_for_plot/ PLOT_PERIOD))
                    #manual save
                    f = open(WEIGHTS_FILE, 'wb')
                    pkl.dump(sess.run(tvars), f,protocol=2)
                    f.close()
                    print('auto-saved weights successfully.')

                #Do the training step
                if(episode_number%PERIOD==0):
                    running_reward += period_reward / PERIOD

                    #choose learning rate:
                    grad_dict = {Gradients_holder[i]: grads_sums[i] for i in range(VAR_NO)}
                    if(previous_period_reward is None or period_reward / PERIOD > previous_period_reward):
                        grad_dict.update({learning_rate: 1e-2})
                        print("Learning rate is now 1e-2")
                    else:
                        grad_dict.update({learning_rate: 1e-4})
                        print("Learning rate is now 1e-4")

                    #take the train step
                    sess.run(train_step, feed_dict=grad_dict)

                    print ('Episode No. %d, Steps No. %d,   Episodes average reward %f., Total average reward %f.' % (episode_number, steps, period_reward / PERIOD, running_reward / (episode_number//PERIOD)))
                    print('Amount of manually drawing this period: %d' %(manual_prob_use))
                    previous_period_reward = period_reward / PERIOD
                    period_reward = 0
                    manual_prob_use = 0
                    grads_sums = get_empty_grads_sums()

                #add reward to plot array
                if(episode_number%PLOT_PERIOD == 0):
                    rewards_per_episode[(episode_number // PLOT_PERIOD)-1] = reward_for_plot / PLOT_PERIOD
                    reward_for_plot = 0

                #nullify relevant vars and updates episode number.
                rewards, states, actions_booleans = [], [], []
                steps = 0
                obsrv = env.reset()

    IS_NORMALIZED = ""
    if(DO_NORMALIZE):
        IS_NORMALIZED = "normalized"
    plt.plot(rewards_per_episode)
    plt.title(ENVIRONMENT+" Rewards Average per "+str(PLOT_PERIOD)+" Episodes for "+IS_NORMALIZED+" NN with "+str(LAYERS_NO)+" layers")
    plt.savefig("..\\..\\graphs\\rewards_for_"+ENVIRONMENT+"_period_"+str(PLOT_PERIOD)+"_layers_"+str(LAYERS_NO)+"_"+IS_NORMALIZED+"_episodes_"+str(TOTAL_EPISODES)+".png")

    print("Finised running after %d episoded! best average reward for %d episodes is %f."% (episode_number,PLOT_PERIOD,save_reward))

if __name__ == '__main__':
    tf.app.run()
