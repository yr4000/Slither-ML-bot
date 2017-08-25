'''
here will be all the functions used by the model
'''

import tensorflow as tf
import numpy as np
import math
import json
import time
from datetime import datetime

DO_NOTHING, MOVE_RIGHT, MOVE_LEFT = 0, 1, 2
SLICES_NO = 32
INNER_INPUT_SIZE = 400

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
    return data["observation"], data["score"], data["is_dead"],data['message_id'], default

#TODO: temporary solution, need to fix
def get_default_data():
    return {
        'observation': [0 for i in range(INNER_INPUT_SIZE)],
        'score': 0,
        'is_dead': False,
        'message_id': -1
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
        'request_id': request_id
    }

#a simple reward function to begin with
def get_reward(score_arr,is_dead):
    no_gain_punishement = -5
    death_punishment = -100
    small_gain_prize = 2
    medium_gain_prize = 6
    high_gain_prize = 10
    rewards = []

    if (len(score_arr) == 1):
        return np.array([0]) # worst case TODO : i think should never happen im model

    scores_diff = np.diff(score_arr)    #convert raw score to points earned/lost per step
    for k in scores_diff:
        if(k<=0):
            rewards.append(no_gain_punishement)
        elif(k>0 and k<=5):
            rewards.append(small_gain_prize)
        elif(k>5 and k<=15):
            rewards.append(medium_gain_prize)
        else:
            rewards.append(high_gain_prize)

    '''
    #boost positive rewards and decay negative once
    for i in range(len(rewards)):
        if(rewards[i] <= 0):
            rewards[i] -= boost_const
        else:
            rewards[i] += boost_const
    '''

    if (is_dead):
        rewards[len(rewards) - 1] = death_punishment

    return rewards

def raw_score_reward(raw_score, is_dead):
    death_punishment = -100
    if (is_dead):
        raw_score[len(raw_score) - 1] = death_punishment

    return raw_score[1:]


def wait_for_game_to_start():
    obsrv, score, is_dead, request_id, default  = get_observation()
    while(is_dead):
        time.sleep(0.5)
        obsrv, score, is_dead, request_id, default = get_observation()



#TODO: this is a good example how to implement switch-case in python. delete in the end
'''
#TODO: temporary
#according to the action, send to the server what to do
def choose_action(index):
    return{
        # do nothing
        0:{
            'action': DO_NOTHING,
            'do_accelerate': False
        },
        #accelerate
        1:{
            'action': DO_NOTHING,
            'do_accelerate': True
        },
        #move right
        2:{
            'action': MOVE_RIGHT,
            'do_accelerate': False
        },
        #move right and accelerate
        3:{
            'action': MOVE_RIGHT,
            'do_accelerate': True
        },
        #move left
        4:{
            'action': MOVE_LEFT,
            'do_accelerate': False
        },
        #move left and accelerate
        5:{
            'action': MOVE_LEFT,
            'do_accelerate': True
        }
    }[index]
'''
