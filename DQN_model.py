'''
This code is based on the following guide (for DQN for pong) and it's code:
http://www.danielslater.net/2016/03/deep-q-learning-pong-with-tensorflow.html
'''
from utils.plot_utils import plot_graph
from utils.net_utils import *
from utils.models_utils import *
from utils.log_utils import Logger
from collections import deque
import pickle as pkl
import os
import random
import time
import numpy as np
import tensorflow as tf


#MODEL CONSTANTS

VAR_NO = DQN_params['VAR_NO']  # number of Ws and bs (the variables)
EPSILON_FOR_EXPLORATION = DQN_params['EPSILON_FOR_EXPLORATION']

# Model constants
MAX_STEPS = DQN_params['MAX_STEPS']
NUM_OF_EPOCHS = DQN_params['NUM_OF_EPOCHS']
NUM_OF_GAMES_FOR_TEST = DQN_params['NUM_OF_GAMES_FOR_TEST']

# Load and save constants
WEIGHTS_FILE = DQN_params['WEIGHTS_FILE']
LOAD_WEIGHTS = DQN_params['LOAD_WEIGHTS']

#game constants:
BEGINING_SCORE = 10

#initialize logger:
logger = Logger('DQN_test')

class Agent:
    # agent's constants:
    #net constants
    LEARN_RATE = DQN_params['LEARN_RATE']

    #variables sizes
    FRAMES_PER_OBSERVATION = DQN_params['FRAMES_PER_OBSERVATION']      #TODO: in the original code it was 4, we need to figure out what he expected to get...
    LAST_RAW_SCORES_SIZE = DQN_params['LAST_RAW_SCORES_SIZE'] #TODO : could be as low as 2 , but to keep a buffer
    MEMORY_SIZE = DQN_params['MEMORY_SIZE']

    #logic constants:
    MIN_MEMORY_SIZE_FOR_TRAINING = DQN_params['MIN_MEMORY_SIZE_FOR_TRAINING']
    MINI_BATCH_SIZE = DQN_params['MINI_BATCH_SIZE']
    FUTURE_REWARD_DISCOUNT = DQN_params['FUTURE_REWARD_DISCOUNT']

    #action constants:
    INITIAL_RANDOM_ACTION_PROB = DQN_params['INITIAL_RANDOM_ACTION_PROB']  # starting chance of an action being random
    FINAL_RANDOM_ACTION_PROB = DQN_params['FINAL_RANDOM_ACTION_PROB']  # final chance of an action being random
    CONST_DECREASE_IN_EXPLORATION = \
        (INITIAL_RANDOM_ACTION_PROB-FINAL_RANDOM_ACTION_PROB)/MAX_STEPS

    #indexes:
    OBS_LAST_STATE_INDEX, OBS_ACTION_INDEX, OBS_REWARD_INDEX, OBS_CURRENT_STATE_INDEX, OBS_TERMINAL_INDEX = range(5)

    # do write to log?
    WRITE_TO_LOG_EVERY = DQN_params['WRITE_TO_LOG_EVERY']

    def __init__(self):
        #variables to train the net
        self.sess = tf.Session()
        self.input_layer, self.output_layer = create_CNN()
        self.actions = tf.placeholder(tf.float32, [None, OUTPUT_DIM])      #TODO: is it good it uses the output dim from the net utils?
        self.targets = tf.placeholder(tf.float32, [None])
        self.tvars = tf.trainable_variables()

        #memory variables
        self.last_state = None      #each state will consist of 4 frames
        self.memory = deque()       #gets tuples of (Xt,At,Rt,X(t+1),terminal)

        #action variables:
        self.probability_of_random_action = self.INITIAL_RANDOM_ACTION_PROB

        # set the first action go up (arbitrary choice)
        self.last_action = np.zeros(OUTPUT_DIM)
        self.last_action[0] = 1

        #this part is to do the train step
        #TODO: is this the correct way to do the train step? tom: i think yes
        readout_action = tf.reduce_sum(tf.multiply(self.output_layer, self.actions), reduction_indices=1)     #TODO: is this suppose to be multiply for sure, but between what?
        cost = tf.reduce_mean(tf.square(self.targets - readout_action))
        self.train_operation = tf.train.AdamOptimizer(self.LEARN_RATE).minimize(cost)

        #initialize variables
        self.sess.run(tf.global_variables_initializer())

        #variables for evaluation:
        self.last_raw_scores = deque()
        self.last_raw_scores.append(BEGINING_SCORE)

        #to count the episodes:
        self.step_number = 0

        # check if file is not empty
        if (os.path.isfile(WEIGHTS_FILE) and LOAD_WEIGHTS):
            self.load_weights()

        # creates file if it doesn't exisits:
        if (not os.path.isfile(WEIGHTS_FILE)):
            open(WEIGHTS_FILE, 'a').close()

    def take_action(self,request_id):
        # TODO : how to decay is always up for debate
        #gradually desrease epsilon, in epsilon greedy policy
        if self.probability_of_random_action > self.FINAL_RANDOM_ACTION_PROB \
                and len(self.memory) > self.MIN_MEMORY_SIZE_FOR_TRAINING:

            self.probability_of_random_action -= self.CONST_DECREASE_IN_EXPLORATION

        #select an action according to the Q function with epsilon greedy
        new_action = np.zeros([OUTPUT_DIM])

        if (random.random() <= self.probability_of_random_action):
            # choose an action randomly
            action_index = random.randrange(OUTPUT_DIM)
        else:
            # choose an action given our last state
            readout_t = self.sess.run(self.output_layer, feed_dict={self.input_layer: [self.last_state]})#TODO: [0]
            if self.step_number % self.WRITE_TO_LOG_EVERY ==0:
                logger.write_to_log("Action Q-Values are {}".format(readout_t))
            action_index = np.argmax(readout_t)

        new_action[action_index] = 1

        self.last_action = new_action

        #write the action down
        send_action(action_index,request_id)

    # take care of the data and store it in the memory
    def take_one_step(self):
        #This part of the code processes the data
        frame, score, is_dead, request_id, default_obsrv = get_observation()  # get observation
        #if the game began we stack the same frame FRAMES_PER_OBSERVATION times
        if self.last_state is None:
            self.last_state = np.stack(tuple(frame for i in range(self.FRAMES_PER_OBSERVATION)))
            return
        #poping out the first frame and pushing the last one
        current_state = np.append(self.last_state[1:], [frame], axis = 0)

        #adding score:
        self.last_raw_scores.append(score)
        if(len(self.last_raw_scores) > self.LAST_RAW_SCORES_SIZE):
            self.last_raw_scores.popleft()

        #getting reward:
        reward = self.get_reward(self.last_raw_scores,is_dead)

        #take an action:
        self.take_action(request_id)

        #adding observarion to memory:
        #TODO: currently, if we get is dead for the current state, then the reward is for the action At. I think it's fine but not sure
        self.memory.append((self.last_state, self.last_action, reward, current_state, is_dead))
        #pop out memory:
        if(len(self.memory) > self.MEMORY_SIZE):
            self.memory.popleft()
        #if enough steps passed - train:
        if(len(self.memory) > self.MIN_MEMORY_SIZE_FOR_TRAINING):
            self.train()

        #if the bot died restart the observation and raw_score_counting
        if(is_dead):
            self.last_state = None
            self.last_raw_scores.clear()
            self.last_raw_scores.append(BEGINING_SCORE)

        self.step_number += 1
        time.sleep(0.05)            #TODO: is this necessary?
        #TODO: make sure
        self.last_state = current_state

    #TODO: check how long this takes, and if there is a better way to do the train (currently it's an exact copy of the origin)
    def train(self):
        # sample a mini_batch to train on
        mini_batch = random.sample(self.memory, self.MINI_BATCH_SIZE)
        # get the batch variables
        previous_states = [d[self.OBS_LAST_STATE_INDEX] for d in mini_batch]
        actions = [d[self.OBS_ACTION_INDEX] for d in mini_batch]
        rewards = [d[self.OBS_REWARD_INDEX] for d in mini_batch]
        current_states = [d[self.OBS_CURRENT_STATE_INDEX] for d in mini_batch]
        agents_expected_reward = []
        # this gives us the agents expected reward for each action we might
        agents_reward_per_action = self.sess.run(self.output_layer, feed_dict={self.input_layer: current_states})
        for i in range(len(mini_batch)):
            if mini_batch[i][self.OBS_TERMINAL_INDEX]:
                # this was a terminal frame so there is no future reward...
                agents_expected_reward.append(rewards[i])
            else:
                agents_expected_reward.append(
                    rewards[i] + self.FUTURE_REWARD_DISCOUNT * np.max(agents_reward_per_action[i]))

        # learn that these actions in these states lead to this reward
        self.sess.run(self.train_operation, feed_dict={
            self.input_layer: previous_states,
            self.actions: actions,
            self.targets: agents_expected_reward})



    def get_reward(self, raw_scores,is_dead):
        death_punishment = -10 #TODO : i think -100 is way too high because we do not normalize rewards
        no_gain_punishment = -0.05
        reward = raw_scores[-1] - raw_scores[-2]
        if(is_dead):
            reward = death_punishment
        elif(reward <= 0):
            reward += no_gain_punishment

        return reward


    def load_weights(self):
        with open(WEIGHTS_FILE, 'rb') as f:  # BEST_WEIGHTS
            for var, val in zip(self.tvars, pkl.load(f)):
                self.sess.run(tf.assign(var, val))

    def save_weights(self):
        with open(WEIGHTS_FILE, 'wb') as f:
            pkl.dump(self.sess.run(self.tvars), f, protocol=2)

    def test(self):
        scores = []
        frame, score, is_dead, request_id, default_obsrv = get_observation()  # get observation
        while(not is_dead):
            frame, score, is_dead, request_id, default_obsrv = get_observation()  # get observation
        #run NUM_OF_GAMES_FOR_TEST of games and avarage their score
        for i in range(NUM_OF_GAMES_FOR_TEST):
            while (is_dead):
                frame, score, is_dead, request_id, default_obsrv = get_observation()  # get observation
            state = np.stack(tuple(frame for i in range(self.FRAMES_PER_OBSERVATION)))
            while (not is_dead):
                #feed forward pass
                readout_t = self.sess.run(self.output_layer, feed_dict={self.input_layer: [state]})#TODO: [0]
                #choose and send action
                send_action(np.argmax(readout_t), request_id)
                #get next observation
                frame, score, is_dead, request_id, default_obsrv = get_observation()
                state = np.append(state[1:], [frame], axis=0)

            if(is_dead):
                scores.append(score)
                state = None

        return(np.average(scores))


if __name__ == '__main__':

    #initialize agent
    avg_scores_per_game = []
    agent = Agent()
    #test first time with random weights for baseline
    avg_scores_per_game.append(agent.test())
    for i in range(1 , NUM_OF_EPOCHS+1):
        if 1/i >= agent.FINAL_RANDOM_ACTION_PROB:
            agent.INITIAL_RANDOM_ACTION_PROB = 1/i

        agent.step_number = 0

        #loop over k steps
        while agent.step_number < MAX_STEPS:
            agent.take_one_step()
        avg_scores_per_game.append(agent.test())
        plot_graph(avg_scores_per_game ,"avg_score_per_epoch" ,"DQN_avg_score_by_epoch" )
