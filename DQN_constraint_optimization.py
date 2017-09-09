'''
This code is based on the following guide (for DQN for pong) and it's code:
http://www.danielslater.net/2016/03/deep-q-learning-pong-with-tensorflow.html
'''
from _lsprof import profiler_entry

from utils.parameters_utils import *        #This line auto generated the parameters if they are not exist
from utils.plot_utils import plot_graph
from utils.models_utils import *
from utils.net_utils import *
from utils.log_utils import Logger
from collections import deque
import pickle as pkl
import os
import random
import time
import numpy as np
import tensorflow as tf


#MODEL CONSTANTS

VAR_NO = DQN_params['VAR_NO']  # number of Ws and bs (the variables), number of layers X 2

# Model constants
MAX_STEPS = DQN_params['MAX_STEPS']
EPISODE_SIZE = DQN_params['EPISODE_SIZE']
NUM_OF_EPOCHS = DQN_params['NUM_OF_EPOCHS']
NUM_OF_GAMES_FOR_TEST = DQN_params['NUM_OF_GAMES_FOR_TEST']

# Load and save constants
WEIGHTS_FILE = DQN_params['WEIGHTS_DIR'] + DQN_params['WEIGHTS_FILE']
BEST_WEIGHTS = DQN_params['WEIGHTS_DIR'] + DQN_params['BEST_WEIGHTS']
DO_LOAD_WEIGHTS = DQN_params['DO_LOAD_WEIGHTS']
WEIGHTS_DIR = DQN_params['WEIGHTS_DIR']
WEIGHTS_TO_LOAD = DQN_params['WEIGHTS_TO_LOAD']

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
    LOCAL_HORIZON =  DQN_params['LOCAL_HORIZON']
    MIN_MEMORY_SIZE_FOR_TRAINING = DQN_params['MIN_MEMORY_SIZE_FOR_TRAINING']
    MINI_BATCH_SIZE = DQN_params['MINI_BATCH_SIZE']
    FUTURE_REWARD_DISCOUNT = DQN_params['FUTURE_REWARD_DISCOUNT']
    PENALTY_COEFF = DQN_params['PENALTY_COEFF']
    #action constants:
    INITIAL_RANDOM_ACTION_PROB = DQN_params['INITIAL_RANDOM_ACTION_PROB']  # starting chance of an action being random
    FINAL_RANDOM_ACTION_PROB = DQN_params['FINAL_RANDOM_ACTION_PROB']  # final chance of an action being random
    CONST_DECREASE_IN_EXPLORATION = \
        (INITIAL_RANDOM_ACTION_PROB-FINAL_RANDOM_ACTION_PROB)/(MAX_STEPS*NUM_OF_EPOCHS)

    #indexes:
    OBS_LAST_STATE_INDEX, OBS_ACTION_INDEX, OBS_REWARD_INDEX,OBS_CUM_REWARD_INDEX, OBS_CURRENT_STATE_INDEX, OBS_TERMINAL_INDEX = range(6)

    # do write to log?
    WRITE_TO_LOG_EVERY = DQN_params['WRITE_TO_LOG_EVERY']

    #test or train mode
    TEST_MODE = DQN_params['TEST_MODE']
    if(TEST_MODE):
        print("Test mode!")
        logger.write_to_log("Test mode!")
    #do learn from expert?
    LEARN_FROM_EXPERT = DQN_params['LEARN_FROM_EXPERT']
    if(LEARN_FROM_EXPERT):
        print("Learn from expert mode!")
        logger.write_to_log("Learn from expert mode!")

    def __init__(self):
        #variables to train the net
        self.sess = tf.Session()
        self.input_layer, self.output_layer = create_CNN()
        self.actions = tf.placeholder(tf.float32, [None, OUTPUT_DIM])      #TODO: is it good it uses the output dim from the net utils?
        self.targets = tf.placeholder(tf.float32, [None])
        self.lower_bounds = tf.placeholder(tf.float32,[None])
        self.upper_bounds = tf.placeholder(tf.float32,[None])
        self.tvars = tf.trainable_variables()

        #memory variables
        self.last_state = None      #each state will consist of 4 frames
        self.memory_buffer = deque() #gets tuples of (Xt,At,rt,None,X(t+1),terminal)
        self.memory = deque()        #gets tuples of (Xt,At,rt,Rt,X(t+1),terminal)

        #action variables:
        self.probability_of_random_action = self.INITIAL_RANDOM_ACTION_PROB

        # set the first action go up (arbitrary choice)
        self.last_action = np.zeros(OUTPUT_DIM)
        self.last_action[0] = 1

        #this part is to do the train step
        #TODO: is this the correct way to do the train step? tom: i think yes
        readout_action = tf.reduce_sum(tf.multiply(self.output_layer, self.actions), reduction_indices=1)     #TODO: is this suppose to be multiply for sure, but between what?
        lower_loss = tf.square(tf.nn.relu(self.lower_bounds - readout_action))
        upper_loss = tf.square(tf.nn.relu(self.upper_bounds - readout_action))
        loss = tf.square(self.targets - readout_action)
        cost = tf.reduce_mean(loss + tf.multiply(self.PENALTY_COEFF , lower_loss) +
                              tf.multiply(self.PENALTY_COEFF, upper_loss))
        self.train_operation = tf.train.AdamOptimizer(self.LEARN_RATE).minimize(cost)

        #initialize variables
        self.sess.run(tf.global_variables_initializer())

        #variables for evaluation:
        self.last_raw_scores = deque()
        self.last_raw_scores.append(BEGINING_SCORE)
        self.bonus = 0

        #time and evaluation variables:
        self.step_number = 0
        self.epoch_no = 0

        #create folder for weights:
        if (not os.path.exists(WEIGHTS_DIR)):
            os.makedirs(WEIGHTS_DIR)

        # check if file is not empty
        if (os.path.isfile(WEIGHTS_TO_LOAD) and DO_LOAD_WEIGHTS):
            self.load_weights(WEIGHTS_TO_LOAD)

        # creates file if it doesn't exisits:
        if (not os.path.isfile(WEIGHTS_FILE)):
            open(WEIGHTS_FILE, 'a').close()

    def take_action(self,request_id):
        # TODO : how to decay is always up for debate
        #gradually desrease epsilon, in epsilon greedy policy
        if((self.probability_of_random_action > self.FINAL_RANDOM_ACTION_PROB)
           and (len(self.memory) > self.MIN_MEMORY_SIZE_FOR_TRAINING)):
            self.probability_of_random_action -= self.CONST_DECREASE_IN_EXPLORATION
            if (self.step_number % self.WRITE_TO_LOG_EVERY == 0):
                logger.write_to_log("probability_of_random_action: " + str(self.probability_of_random_action))

        #select an action according to the Q function with epsilon greedy
        new_action = np.zeros([OUTPUT_DIM])

        if (random.random() <= self.probability_of_random_action)and (not self.TEST_MODE):
            # choose an action randomly
            action_index = random.randrange(OUTPUT_DIM)
        else:
            # choose an action given our last state
            readout_t = self.sess.run(self.output_layer, feed_dict={self.input_layer: [self.last_state]})#TODO: [0]
            if(self.step_number % self.WRITE_TO_LOG_EVERY ==0):
                logger.write_to_log("Action Q-Values are {}".format(readout_t))
            action_index = np.argmax(readout_t)

        new_action[action_index] = 1

        self.last_action = new_action

        #write the action down
        send_action(action_index,request_id)

    # take care of the data and store it in the memory
    def take_one_step(self):
        #This part of the code processes the data
        frame, score, self.bonus, is_dead, request_id, default_obsrv, AI_action, AI_accel = get_observation()  # get observation

        #handaling edge cases:
        #connection problems:
        if(score == 0):
            score = self.last_raw_scores[-1]
        #if from some reason the frame we observed is wrong
        if(len(frame) != CNN_params['INPUT_DIM']):
            print("Warning: current frame doesn't have the right dim! skipping this step.")
            print("frame: " + str(frame))
            logger.write_to_log("Warning: current frame doesn't have the right dim! skipping this step.")
            logger.write_to_log("frame: " + str(frame))
            return

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

        #write to log and evaluate
        if(self.step_number % self.WRITE_TO_LOG_EVERY == 0):
            logger.write_to_log("Last raw scores: " + str(self.last_raw_scores))
            print("Avarage raw score for epoch %d step %d: %0.2f" % (self.epoch_no,self.step_number,np.average(self.last_raw_scores)))

        #getting reward:
        reward = self.get_reward(self.last_raw_scores,is_dead)

        if(self.LEARN_FROM_EXPERT):
            self.last_action = make_one_hot(AI_action, AI_accel)

        else:
            self.take_action(request_id)

        #do not train when testing the model
        if(not self.TEST_MODE):
            #insert observation into memory_buffer
            self.memory_buffer.append((self.last_state , self.last_action ,reward, None,
                                      current_state, is_dead))
            #if enough steps passed - train:
            if(len(self.memory) > self.MIN_MEMORY_SIZE_FOR_TRAINING):
                self.train()

        #if the bot died restart the observation and raw_score_counting
        if(is_dead or self.step_number%EPISODE_SIZE == 0):#TODO: think long and hard
            self.move_buffer_to_memory()
        if(is_dead):
            print("just died!")
            logger.write_to_log("just died!")
            self.last_state = None
            self.last_raw_scores.clear()
            self.last_raw_scores.append(BEGINING_SCORE)

        #update last state seen
        self.last_state = current_state

        #write to logger
        if(self.step_number % self.WRITE_TO_LOG_EVERY == 0):
            logger.write_spacer()

        #in the first epoch it runs too fast in relation to the game, so the agent can't really collect observations like this.
        if(len(self.memory) < self.MIN_MEMORY_SIZE_FOR_TRAINING):
            time.sleep(0.3)

        self.step_number += 1

        if(is_dead):
            wait_for_game_to_start()

    #for a mini_batch of 100 train takes about 0.25 seconds#TODO:check the time
    def train(self):
        print("in train!")
        start_time = time.time()
        #make sure we have K steps back and forward
        eligable_obs = list(enumerate(list(self.memory)))[(self.LOCAL_HORIZON+1) * ORIENTATION_VARIATIONS :
                            len(self.memory) - ((self.LOCAL_HORIZON+1) * ORIENTATION_VARIATIONS)]

        # sample a mini_batch to train on
        mini_batch_enamurated = random.sample(eligable_obs , self.MINI_BATCH_SIZE)

        mini_batch = list(list(zip(*mini_batch_enamurated))[1])

        # get the mini_batch variables
        previous_states = [d[self.OBS_LAST_STATE_INDEX] for d in mini_batch]
        actions = [d[self.OBS_ACTION_INDEX] for d in mini_batch]
        rewards = [d[self.OBS_REWARD_INDEX] for d in mini_batch]
        #rewards = normalize_rewards_by_max(rewards) TODO: what to do?
        current_states = [d[self.OBS_CURRENT_STATE_INDEX] for d in mini_batch]
        cum_rewards = [d[self.OBS_CUM_REWARD_INDEX] for d in mini_batch]

        #for each observation calc and Q(Sj,Aj)and upper\lower bounds
        lower_bounds = []
        upper_bounds = []
        agents_expected_reward = []
        agents_reward_per_action = self.sess.run(self.output_layer,
                                                 feed_dict={self.input_layer: current_states})
        for j,obs in enumerate(mini_batch_enamurated):
            # get the adjacant observations for upper and lower bounds
            mini_batch_for_upper_bound = self.get_minibatch_upper_bound(list(obs)[0])
            mini_batch_for_lower_bound = self.get_minibatch_lower_bound(list(obs)[0])

            agents_expected_reward.append(self.calc_agent_expected_reward(
                mini_batch,j,current_states,cum_rewards,rewards,agents_reward_per_action))
            lower_bounds.append(self.calc_lower_bounds(mini_batch_for_lower_bound))
            upper_bounds.append(self.calc_upper_bounds(mini_batch_for_upper_bound))

        if(self.step_number % self.WRITE_TO_LOG_EVERY == 0):
            logger.write_to_log("rewards: " + str(rewards))
            logger.write_to_log("agents_expected_reward: " + str(agents_expected_reward))   #TODO: do we update right?

        #backpropogation
        self.sess.run(self.train_operation, feed_dict={
            self.input_layer: previous_states,
            self.actions: actions,
            self.targets: agents_expected_reward,
            self.lower_bounds:lower_bounds,
            self.upper_bounds: upper_bounds})

        print("the training took {} time to run".format(time.time() - start_time))

    def get_minibatch_upper_bound(self,j):
        l = list(self.memory)[j - ((self.LOCAL_HORIZON+1)*ORIENTATION_VARIATIONS) : j+1]
        mask = np.array(range((self.LOCAL_HORIZON+1)*ORIENTATION_VARIATIONS + 1))% ORIENTATION_VARIATIONS == 0
        batch = []
        for i,obs in enumerate(l):
            if(mask[i]):
                batch.append(obs)
        return batch

    def get_minibatch_lower_bound(self,j):
        l = list(self.memory)[j : j + 1 + ((self.LOCAL_HORIZON+1)*ORIENTATION_VARIATIONS)]
        mask = np.array(range((self.LOCAL_HORIZON+1)*ORIENTATION_VARIATIONS + 1))% ORIENTATION_VARIATIONS == 0
        batch = []
        for i, obs in enumerate(l):
            if(mask[i]):
                batch.append(obs)
        return batch

    def calc_agent_expected_reward(self,mini_batch,j,current_states,cum_rewards,rewards,agents_reward_per_action):
        if mini_batch[j][self.OBS_TERMINAL_INDEX]:
            # this was a terminal frame so there is no future reward...
            return (rewards[j])
        else:
            return(rewards[j] + self.FUTURE_REWARD_DISCOUNT * np.max(agents_reward_per_action[j]))

    def calc_lower_bounds(self,batch):
        #get the variables of the batch
        cum_rewards = np.array([d[self.OBS_CUM_REWARD_INDEX] for d in batch])
        current_states = np.array([d[self.OBS_CURRENT_STATE_INDEX] for d in batch])

        #calculate L_max_j
        discount_factors_1 = np.power(self.FUTURE_REWARD_DISCOUNT, range(1, self.LOCAL_HORIZON + 1))
        r = np.multiply(cum_rewards[2:], discount_factors_1)
        r = np.add(cum_rewards[0],-r)
        agents_reward_per_action = self.sess.run(self.output_layer,feed_dict=
                                {self.input_layer: current_states[2:]})
        #TODO:check
        agents_max_reward_over_action = tf.reduce_max(agents_reward_per_action,reduction_indices=1)
        discount_factors_2 = np.power(self.FUTURE_REWARD_DISCOUNT, range(2, self.LOCAL_HORIZON + 2))
        lower_bounds = r - tf.multiply(agents_max_reward_over_action,discount_factors_2)
        return(self.sess.run(tf.reduce_max(lower_bounds)))

    def calc_upper_bounds(self,batch):
        #get the variables of the batch
        cum_rewards = np.array([d[self.OBS_CUM_REWARD_INDEX] for d in batch])
        current_states = np.array([d[self.OBS_CURRENT_STATE_INDEX] for d in batch])
        actions = np.array([d[self.OBS_ACTION_INDEX] for d in batch])

        discount_factors_1 = np.power(self.FUTURE_REWARD_DISCOUNT ,(range(-self.LOCAL_HORIZON-1, -1)))
        r = np.multiply(cum_rewards[:-2],discount_factors_1)
        r = np.add(r,-cum_rewards[-1])
        agents_reward_per_action = self.sess.run(self.output_layer,feed_dict=
                                {self.input_layer: current_states[:-2]})
        agents_reward_per_action = tf.reduce_sum(tf.multiply
                                (agents_reward_per_action, actions[:-2]), reduction_indices=1)
        discount_factors_2 = np.power(self.FUTURE_REWARD_DISCOUNT, (range(-self.LOCAL_HORIZON - 1, -1)))
        upper_bounds = tf.multiply(agents_reward_per_action,discount_factors_2) - r
        return(self.sess.run(tf.reduce_min(upper_bounds)))

    def move_buffer_to_memory(self):

        print("in move!")
        #calc the cumulative discounted rewards for the episode:
        rewards = list(list(zip(* self.memory_buffer))[self.OBS_REWARD_INDEX])
        cum_rewards = self.discount_rewards(rewards)

        for i,OBSt in enumerate(self.memory_buffer):
            # make St,At,St+1 invariant to oreintation
            SAS_list = \
                make_invariant_to_orientation(OBSt[self.OBS_LAST_STATE_INDEX],
                                              OBSt[self.OBS_ACTION_INDEX],
                                              OBSt[self.OBS_CURRENT_STATE_INDEX][-1])

            # adding observations to memory:
            for sas in SAS_list:
                # TODO: currently, if we get is dead for the current state, then the reward is for the action At. I think it's fine but not sure
                self.memory.append((sas[0], sas[1], OBSt[self.OBS_REWARD_INDEX]
                                    ,cum_rewards[i], sas[2], OBSt[self.OBS_TERMINAL_INDEX]))
                # pop out memory:
                if (len(self.memory) > self.MEMORY_SIZE):
                    self.memory.popleft()

        #clear memory_buffer
        self.memory_buffer.clear()

    def get_reward(self, raw_scores,is_dead):
        death_punishment = -50
        no_gain_punishment = -1
        reward = raw_scores[-1] - raw_scores[-2]
        if(is_dead):
            reward = death_punishment
        elif(reward <= 0):
            reward += no_gain_punishment

        return reward + self.bonus

    #TODO:maybe use numpy?
    def discount_rewards(self, rewards):
        # compute discounted rewards
        cumulative_discounted_rewards = np.zeros(len(rewards))
        partial_sum = 0.0
        for i in reversed(range(0, len(rewards))):
            partial_sum = partial_sum * self.FUTURE_REWARD_DISCOUNT + rewards[i]
            cumulative_discounted_rewards[i] = partial_sum
        return cumulative_discounted_rewards

    def load_weights(self,file_name):
        with open(file_name, 'rb') as f:  # BEST_WEIGHTS
            for var, val in zip(self.tvars, pkl.load(f)):
                self.sess.run(tf.assign(var, val))
        print("successfully loaded " + file_name)

    def save_weights(self,file_name):
        with open(file_name, 'wb') as f:
            pkl.dump(self.sess.run(self.tvars), f, protocol=2)
        print("successfully saved " + file_name)

    def evaluate(self):
        print("started evaulation over {} games".format(NUM_OF_GAMES_FOR_TEST))
        scores = []
        frame, score, bonus, is_dead, request_id, default_obsrv, AI_action, AI_accel = get_observation()  # get observation
        #force an ending of the game.
        #problem: in current implementation the bot dies twice...
        if(not is_dead):
            commit_sucide()
            time.sleep(0.25)
        #run NUM_OF_GAMES_FOR_TEST of games and avarage their score
        for i in range(NUM_OF_GAMES_FOR_TEST):
            wait_for_game_to_start()
            frame, score, bonus, is_dead, request_id, default_obsrv, AI_action, AI_accel = get_observation()  # get observation
            state = np.stack(tuple(frame for i in range(self.FRAMES_PER_OBSERVATION)))
            while (not is_dead):
                #feed forward pass
                readout_t = self.sess.run(self.output_layer, feed_dict={self.input_layer: [state]})#TODO: [0]
                #choose and send action
                send_action(np.argmax(readout_t), request_id)
                #get next observation
                frame, score, bonus, is_dead, request_id, default_obsrv, AI_action, AI_accel = get_observation()
                state = np.append(state[1:], [frame], axis=0)
            print("just died!")
            scores.append(score)
            state = None

        print("finished evaluation.")
        return(np.average(scores))

if __name__ == '__main__':
    #initialize agent
    avg_scores_per_step = []
    avg_scores_per_epoch = []
    best_avg_per_step = 0
    agent = Agent()
    #test first time with random weights for baseline
    #print("evaluationg random movements")
    #avg_scores_per_game.append(agent.evaluate())
    #the division of steps to epochs is for evaluation
    print("experiment started!")

    while(agent.epoch_no < NUM_OF_EPOCHS):
        #nulify relevant properties
        agent.step_number = 0
        avg_scores_per_step = []
        #loop over k steps
        while agent.step_number < MAX_STEPS:
            agent.take_one_step()
            #write to log and add to plot array:
            if (agent.step_number % agent.WRITE_TO_LOG_EVERY == 0):
                avg_scores_per_step.append(np.average(agent.last_raw_scores))
                logger.write_to_log("avg_scores_per_step" + str(avg_scores_per_step))

            #TODO: is that sleep necessary??
            if(agent.LEARN_FROM_EXPERT and agent.epoch_no == 0):
                time.sleep(0.025)        #when learn from expert it runs really reallt fast at the first epoch
        #avg_scores_per_game.append(agent.evaluate())

        #save weights and best weights:
        #TODO: last loaded weights will be overrizde on the first save
        #TODO: posible solution: change file names, save best score in parameters...
        if(not agent.TEST_MODE):
            agent.save_weights(WEIGHTS_FILE)
            if(avg_scores_per_step[-1] > best_avg_per_step):
                best_avg_per_step = avg_scores_per_step[-1]
                agent.save_weights(BEST_WEIGHTS)

        #evaluation and plotting
        plot_graph(avg_scores_per_step ,"Average Score Per 100 Steps" ,"DQN_avg_score_per_step_by_epoch_"+str(agent.epoch_no),"Step No.", "Average score")
        avg_scores_per_epoch.append(np.average(avg_scores_per_step))
        logger.write_to_log("avg_scores_per_epoch" + str(avg_scores_per_epoch))
        logger.write_spacer()
        agent.epoch_no += 1

    plot_graph(avg_scores_per_epoch,"Average Score Per Epoch", "DQN_avg_score_per_epoch_for_experiment","Epoch No.", "Average score")
    print("finished experiement")
