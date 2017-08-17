'''
here will be all the functions used by the model
'''

import tensorflow as tf
import numpy as np
import math
import json

DO_NOTHING, MOVE_RIGHT, MOVE_LEFT = 0,1,2
SLICES_NO = 32

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
    except:
        data = get_default_data()
    return data["observation"], data["score"], data["is_dead"]

#TODO: temporary solution, need to fix
def get_default_data():
    return {
        'observation': [0 for i in range(400)],
        'score': 0,
        'is_dead': False,
    }

#TODO: in case of failure send boolean
def send_action(index):
    action = choose_action(index)
    with open('action.json', 'w') as outfile:
        json.dump(action, outfile)
        outfile.close()

    return True

#input: 0 <= index < 2*SLICES_NO
#the cast to int is needed because numpy types can't be converted to json:
#https://stackoverflow.com/questions/11942364/typeerror-integer-is-not-json-serializable-when-serializing-json-in-python/11942689#11942689
#output: action: the slice the bot will move towards, do_accelerate: 0 for no, 1 for yes
def choose_action(index):
    return {
        'action': int(index%SLICES_NO),
        'do_accelerate': int(index//SLICES_NO)
    }

#a simple reward function to begin with
def get_reward(score_arr,is_dead):
    if(is_dead):
        reward = -100
    else:
        reward = score_arr[-1] - score_arr[-2]

    return reward


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
