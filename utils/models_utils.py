'''
here will be all the functions used by the model
'''

import tensorflow as tf
import numpy as np
import math
import json
import time
from datetime import datetime

#when doing unit test parameter loading fails
'''
SLICES_NO = 32
FRAMES_PER_OBSERVATION = 4
OUTPUT_DIM = 64
INNER_INPUT_SIZE = 9
'''
#load parameters:
with open('parameters/Policy_Gradient_Params.json') as json_data:
    PG_params = json.load(json_data)

with open('parameters/DQN_Params.json') as json_data:
    DQN_params = json.load(json_data)


DO_NOTHING, MOVE_RIGHT, MOVE_LEFT = 0, 1, 2
SLICES_NO = PG_params['SLICES_NO']
OUTPUT_DIM = PG_params['OUTPUT_DIM']
INNER_INPUT_SIZE = PG_params['INPUT_DIM']
SQRT_INPUT_DIM = int(INNER_INPUT_SIZE**0.5)

FRAMES_PER_OBSERVATION = DQN_params['FRAMES_PER_OBSERVATION']

def make_one_hot(index, do_accel):
    new_action = np.zeros([OUTPUT_DIM]).astype(int)
    action_index = index + do_accel *SLICES_NO
    new_action[action_index] = 1
    return new_action


#TODO: is it fine that this function is here?
#TODO: fix according to Carmels version
def get_empty_grads_sums():
    grads_sums = tf.trainable_variables()
    for i, val in enumerate(grads_sums):
        grads_sums[i] = 0
    return grads_sums

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

def get_observation():
    try:
        with open('observation.json') as json_data:
            data = json.load(json_data)
        default = 0
    except:
        #print("got default data")      #todo: delte
        data = get_default_data()
        default = 1
    return data["observation"], data["score"], data["is_dead"],data['message_id'], default, data['AI_direction'], data['AI_Acceleration']

#TODO: temporary solution, need to fix
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
        'AI_Acceleration': 0
        }

#TODO: in case of failure send boolean
def send_action(index, request_id):
    action = choose_action(index,request_id)
    with open('action.json', 'w') as outfile:
        json.dump(action, outfile)
    return True

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
        return np.array([0]) # worst case TODO : i think should never happen im model

    rewards = np.diff(score_arr).astype(np.float32)    #convert raw score to points earned/lost per step

    #boost positive rewards and decay negative once
    for i in range(len(rewards)):
        if(rewards[i] <= 0):
            rewards[i] -= no_gain_punishment

    if (is_dead):
        rewards[-1] = death_punishment

    return rewards

def raw_score_reward(raw_score, is_dead):
    death_punishment = -100

    #punish if reward didn't change
    if (is_dead):
        raw_score[-1] = death_punishment

    return raw_score[1:]

def check_if_died(previous_score, current_score):
    delta = 25      #TODO: arbitrary value
    is_dead = previous_score - current_score > delta
    if(is_dead):
        print("I think the bot died and we missed it.")
        print("current score: " + str(current_score) + ", previous score: " + str(previous_score))
    return is_dead

def wait_for_game_to_start():
    obsrv, score, is_dead, request_id, default, AI_action, AI_accel  = get_observation()
    while(is_dead):
        time.sleep(0.5)
        obsrv, score, is_dead, request_id, default, AI_action, AI_accel = get_observation()

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

#if from some reason the the connection got lost, stops the game
#TODO: what if the bot just went on in an empty area? technically it should change from time to time because the bot itself
def wait_if_connection_lost(observations):
    look_back = 10
    do_write = True
    if(not len(observations) < look_back):
        last_seen = observations[-1]
        for i in range(look_back):
            if(last_seen != observations[-i-1]):
                do_write = False

        if(do_write):
            data = {
                'message_id': -1,
                'observation': last_seen,
                'is_dead': True,
                'score': 10,
                'r': 100,
                'x': -1,
                'y': -1,
                'hours': -1,
                'minutes': -1,
                'seconds': -1,
                'AI_direction': 0,
                'AI_Acceleration': 0

            }
            with open('observation.json', 'w') as outfile:
                json.dump(data, outfile)

            print("I think the connection was lost...")
            wait_for_game_to_start()

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

'''
if __name__ == "__main__":
    s_0 = [list(range(9)) for i in range(FRAMES_PER_OBSERVATION)]
    a = make_one_hot(3,1)
    f = list(range(9,18))
    SAS = make_invariant_to_orientation(s_0,a,f)
    for sas in SAS:
        print("new operation")
        print("s_0 is:")
        print(sas[0])
        print("a is:")
        print(sas[1])
        print("s_1 is:")
        print(sas[2])
'''