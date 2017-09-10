'''
here will be all the functions used by the models
'''
#from utils.plot_utils import obsrv_to_image

import tensorflow as tf
import numpy as np
import math
import json
import time
from datetime import datetime

#load parameters:
with open('parameters/Policy_Gradient_Params.json') as json_data:
    PG_params = json.load(json_data)

with open('parameters/DQN_Params.json') as json_data:
    parameters = json.load(json_data)


SLICES_NO = PG_params['SLICES_NO']
OUTPUT_DIM = PG_params['OUTPUT_DIM']
INNER_INPUT_SIZE = PG_params['INPUT_DIM']
SQRT_INPUT_DIM = int(INNER_INPUT_SIZE**0.5)
FRAMES_PER_OBSERVATION = parameters['FRAMES_PER_OBSERVATION']
ORIENTATION_VARIATIONS = 8

#Input: chosen direction from [0,OUTPUT_DIM/2], do_accel from [0,1]
#Output: one hot vector representing the chosen action
def make_one_hot(index, do_accel):
    new_action = np.zeros([OUTPUT_DIM]).astype(int)
    action_index = index + do_accel *SLICES_NO
    new_action[action_index] = 1
    return new_action

#TODO: This function works fine but it's not accurate since it returns a [0]*VAR_NO array instead zero array in the trainable variables dimension. We abandoned this function since we stop working with it.
def get_empty_grads_sums():
    grads_sums = tf.trainable_variables()
    for i, val in enumerate(grads_sums):
        grads_sums[i] = 0
    return grads_sums

#This function is a manual implementation of np.random.multinomial,
#since the last function may crush if the vector it gets sums to a value greater than 1.
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

#picks action randomly from a uniform distribution.
def pick_action_uniformly(actions):
    r = 1
    while(r==1):
        r = np.random.uniform()
    l = len(actions)
    index = math.floor(r*l)
    res = [0 if i != index else 1 for i in range(l)]
    return res

#we assume here we get the array in the right order, so each sum is indeed being multiply with the right factor
def decrese_rewards(rewards):
    gama = 0.99
    dec_arr = np.array([gama**(len(rewards)-t) for t in range(len(rewards))])
    res = np.multiply(rewards,dec_arr)
    return res

def normalize_rewards_by_max(rewards):
    return np.array(rewards)/np.max(np.abs(rewards))

#get's the observation written to the file
def get_observation():
    try:
        with open('observation.json') as json_data:
            data = json.load(json_data)
        default = 0
    except:
        data = get_default_data()
        default = 1
    return data["observation"], data["score"], data["bonus"], data["is_dead"],data['message_id'], default, data['AI_direction'], data['AI_Acceleration']

#in case the function above failed to read from the file
def get_default_data():
    return {
        'observation': [0 for i in range(INNER_INPUT_SIZE)],
        'score': 0,
        'is_dead': False,
        'message_id': -1,
        'currentBotDirection': 0,
        'currentBotAcceleration': 0,
        'hours': -1,
        'minutes': -1,
        'seconds': -1,
        'AI_direction': 0,
        'AI_Acceleration': 0,
        'bonus': 0
        }

def send_action(index, request_id):
    action = choose_action(index,request_id)
    with open('action.json', 'w') as outfile:
        json.dump(action, outfile)

#input: 0 <= index < 2*SLICES_NO
#the cast to int is needed because numpy types can't be converted to json:
#https://stackoverflow.com/questions/11942364/typeerror-integer-is-not-json-serializable-when-serializing-json-in-python/11942689#11942689
#output: action: the slice the bot will move towards, do_accelerate: 0 for no, 1 for yes
def choose_action(index,request_id):
    time = datetime.now().time()
    return {
        'action': int(index%SLICES_NO),
        'do_accelerate': int(index//SLICES_NO),
        'request_id': request_id,
        'commit_sucide': False
    }

#a simple reward function to begin with
def get_reward(score_arr,is_dead):
    no_gain_punishment = 0.05
    death_punishment = -100

    if(len(score_arr) == 1):
        return np.array([0]) # worst case

    rewards = np.diff(score_arr).astype(np.float32)    #convert raw score to points earned/lost per step

    #boost positive rewards and decay negative once
    for i in range(len(rewards)):
        if(rewards[i] <= 0):
            rewards[i] -= no_gain_punishment

    if (is_dead):
        rewards[-1] = death_punishment

    return rewards

#This function seems to work but also doesn't completely solves the missing deaths issue.
def check_if_died(previous_score, current_score):
    delta = 25      #arbitrary value
    is_dead = previous_score - current_score > delta
    if(is_dead):
        print("I think the bot died and we missed it.")
        print("current score: " + str(current_score) + ", previous score: " + str(previous_score))
    return is_dead

#If the bot died this function waits for a new game to start.
def wait_for_game_to_start():
    obsrv, score, bonus, is_dead, request_id, default, AI_action, AI_accel  = get_observation()
    while(is_dead):
        time.sleep(0.5)
        obsrv, score, bonus, is_dead, request_id, default, AI_action, AI_accel = get_observation()

#if you want to force the game to end you can use this function.
def commit_sucide():
    action = {
        'action': 0,
        'do_accelerate': 0,
        'request_id': -1,
        'commit_sucide': True
    }

    with open('action.json', 'w') as outfile:
        json.dump(action, outfile)
    return True

#This function genereates additional observation from the original observation, to augment the possible data the model can learn from.
def generate_alternate_states(state,number_of_frames):
    #turn frames from vectors to matrices
    state_as_matrices = \
        [np.array(frame).reshape(SQRT_INPUT_DIM, SQRT_INPUT_DIM) for frame in state]
    #generate alternate states
    state_list = []
    for i in range(number_of_frames):
        frame_list = []
        curr_frame = state_as_matrices[i]

        frame_list.append(curr_frame)
        frame_list.append(np.rot90(curr_frame))
        frame_list.append(np.rot90(curr_frame, 2))
        frame_list.append(np.rot90(curr_frame, 3))
        frame_list.append(np.flip(curr_frame, 0))
        frame_list.append(np.flip(curr_frame, 1))
        frame_list.append(curr_frame.transpose())
        frame_list.append(np.flip(np.flip(curr_frame, 0), 1).transpose())

        #turn alternate frames from matrices to vectors
        state_list.append([f.reshape([-1]) for f in frame_list])

    state_list = list(zip(*state_list))
    state_list = [list(t)for t in state_list]
    return state_list

#This function generates actions from the original chosen action, compatible to the additional observations above
def generate_alternate_actions(action_index):
    action_list = []
    action_list.append(action_index)
    action_list.append((action_index - SLICES_NO/4) %SLICES_NO)
    action_list.append((action_index - 2*SLICES_NO/4) %SLICES_NO)
    action_list.append((action_index - 3*SLICES_NO/4) %SLICES_NO)
    action_list.append((2*SLICES_NO/4 - action_index) %SLICES_NO)
    action_list.append((0*SLICES_NO/4 - action_index) %SLICES_NO)
    action_list.append((3*SLICES_NO/4 - action_index) % SLICES_NO)
    action_list.append((SLICES_NO/4 - action_index) % SLICES_NO)
    return action_list

#Input: previous state, chosen action, current frame
#Output: list with the original transition and 7 additional transitions the model can learn from
def make_invariant_to_orientation(prev_state, action, curr_frame):
    #generate all alternate orientation states
    St = generate_alternate_states(prev_state,FRAMES_PER_OBSERVATION)
    Ft = generate_alternate_states([curr_frame],1)
    # generte St+1 from St and Ft
    St_1 = [St[i][1:]+Ft[i] for i in range (len(St))]

    #generate alternate actions
    action_index = np.argmax(action) % SLICES_NO
    do_accel = int(np.argmax(action) >= SLICES_NO)

    At = [make_one_hot(int(action_i),do_accel)
                   for action_i in generate_alternate_actions(action_index)]

    #zip together
    SAS_list = list(zip(St,At,St_1))
    return(SAS_list)

#unitest for make_invariant_to_orientation
if __name__ == "__main__":
    frame, score, bonus, is_dead, request_id, default, AI_action, AI_accel = get_observation()
    state = np.stack(tuple(frame for i in range(FRAMES_PER_OBSERVATION)))
    for i in range(3):
        frame, score, bonus, is_dead, request_id, default, AI_action, AI_accel = get_observation()
        state = np.append(state[1:], [frame], axis=0)
        time.sleep(0.7)

    frame, score, bonus, is_dead, request_id, default, AI_action, AI_accel = get_observation()

    action = math.floor(np.random.uniform()*OUTPUT_DIM)
    action = make_one_hot(action%32, action//32)

    sas = make_invariant_to_orientation(state,action,frame)

    for i in range(len(sas)):
        for j in range(len(sas[i][2])):
            matrix_obsrv = np.array(sas[i][2][j]).reshape(SQRT_INPUT_DIM, SQRT_INPUT_DIM)
            #obsrv_to_image(matrix_obsrv,"vers"+str(i)+"."+str(j))

    actions = [np.argmax(sas[i][1])%32 for i in range(len(sas))]
    print(actions)