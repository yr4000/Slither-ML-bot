'''
This code is based on the following guide (for DQN for pong) and it's code:
http://www.danielslater.net/2016/03/deep-q-learning-pong-with-tensorflow.html
'''

from utils.net_utils import *
from utils.models_utils import *
from utils.log_utils import Logger
from collections import deque
import pickle as pkl
import os
import random
import time


#MODEL CONSTANTS

VAR_NO = 12  # number of Ws and bs (the variables)
EPSILON_FOR_EXPLORATION = 0.01

# Model constants
MAX_EPISODES = 1000000


# Load and save constants
WEIGHTS_FILE = 'weights.pkl'
BEST_WEIGHTS = 'best_weights.pkl'
LOAD_WEIGHTS = False

#game constants:
BEGINING_SCORE = 10

#initialize logger:
logger = Logger()

class Agent:
    # agent's constants:
    #net constants
    LEARN_RATE = 1e-6

    #variables sizes
    FRAMES_PER_OBSERVATION = 1      #TODO: in the original code it was 4, we need to figure out what he expected to get...
    LAST_RAW_SCORES_SIZE = 200
    MEMORY_SIZE = 500000

    #logic constants:
    OBSERVATION_STEPS = 50000
    MINI_BATCH_SIZE = 100
    FUTURE_REWARD_DISCOUNT = 0.99

    #action constants:
    INITIAL_RANDOM_ACTION_PROB = 0.5  # starting chance of an action being random   #TODO: should be 1
    FINAL_RANDOM_ACTION_PROB = 0.05  # final chance of an action being random

    #indexes:
    OBS_LAST_STATE_INDEX, OBS_ACTION_INDEX, OBS_REWARD_INDEX, OBS_CURRENT_STATE_INDEX, OBS_TERMINAL_INDEX = range(5)

    # do write to log?
    WRITE_TO_LOG = True

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
        #TODO: is this the correct way to do the train step?
        readout_action = tf.reduce_sum(tf.multiply(self.output_layer, self.actions), reduction_indices=1)     #TODO: is this suppose to be multiply for sure, but between what?
        cost = tf.reduce_mean(tf.square(self.targets - readout_action))
        self.train_operation = tf.train.AdamOptimizer(self.LEARN_RATE).minimize(cost)

        #initialize variables
        self.sess.run(tf.global_variables_initializer())

        #variables for evaluation:
        self.last_raw_scores = deque()
        self.last_raw_scores.append(BEGINING_SCORE)

        #to count the episodes:
        self.episode_number = 0

        # check if file is not empty
        if (os.path.isfile(WEIGHTS_FILE) and LOAD_WEIGHTS):
            self.load_weights()

        # creates file if it doesn't exisits:
        if (not os.path.isfile(WEIGHTS_FILE)):
            open(WEIGHTS_FILE, 'a').close()
        if (not os.path.isfile(BEST_WEIGHTS)):
            open(BEST_WEIGHTS, 'a').close()
            print("created weights file sucessfully!")

    def take_action(self,request_id):
        #if enough episodes passed - start decreasing epsilon   #TODO: COMPLETE!!!

        #select an action according to the Q function with epsilon greedy
        new_action = np.zeros([OUTPUT_DIM])

        if (random.random() <= self.probability_of_random_action):
            # choose an action randomly
            action_index = random.randrange(OUTPUT_DIM)
        else:
            # choose an action given our last state
            readout_t = self.sess.run(self.output_layer, feed_dict={self.input_layer: [self.last_state]})[0]
            if(self.WRITE_TO_LOG):
                logger.write_to_log("Action Q-Values are %s" % readout_t)
            action_index = np.argmax(readout_t)

        new_action[action_index] = 1

        self.last_action = new_action

        #write the action down
        send_action(action_index,request_id)

    # take care of the data and store it in the memory
    def play(self):
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
        if(len(self.memory) > self.OBSERVATION_STEPS):
            self.train()

        #if the bot died restart the observation and raw_score_counting
        if(is_dead):
            self.last_state = None
            self.last_raw_scores.clear()
            self.last_raw_scores.append(BEGINING_SCORE)

        self.episode_number += 1
        time.sleep(0.05)            #TODO: is this necessary?


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

        #TODO: if some condition - save weights - COMPLETE


    def get_reward(self, raw_scores,is_dead):
        death_punishment = -100
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
        with open(BEST_WEIGHTS, 'wb') as f:
            pkl.dump(self.sess.run(self.tvars), f, protocol=2)


if __name__ == '__main__':
    #initialize agent
    agent = Agent()
    #loop over k episodes
    while agent.episode_number < MAX_EPISODES:
        agent.play()