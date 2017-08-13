'''
here will be all the functions used by the model
'''

import tensorflow as tf
import numpy as np
import math
import json

DO_NOTHING, MOVE_RIGHT, MOVE_LEFT = 0,1,2

#TODO: is it fine that this function is here?
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

def get_observation():
    try:
        with open('observation.json') as json_data:
            data = json.load(json_data)
    except:
        data = get_default_data()
    print(data)
    return data["observation"], data["score"], data["game_over"], data["update_weights"]

#TODO: temporary solution, need to fix
def get_default_data():
    return {
        'observation': [0 for i in range(400)],
        'score': 0,
        'game_over': False,
        'update_weights': False
    }


#TODO: in case of failure send boolean
def send_action(index):
    action = choose_action(index)
    with open('action.json', 'w') as outfile:
        json.dump(action, outfile)
        outfile.close()

    return True

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

#TODO: Those functions should be on the JS because they change the position,
#TODO: so in case there is a delay it will keep do the action and not get stuck
#TODO: generaly, all position calculations should be done on the client
def move_right():
    pass

def move_left():
    pass