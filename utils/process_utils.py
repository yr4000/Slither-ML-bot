'''
This file will contain all the functions that process the incformation
in order to turn it to a vector which will be the input of the model
'''

import numpy as np

'''
Input: a snakes object (dict)
Output: a vector which contains significant data about the snakes
'''
#TODO: version1: just take from each snake all it's points
def process_snakes(snakes):
    res = []
    for i in range(1,len(snakes)):  #snakes[0] is the player itself
        for point in snakes[i]['pts']:
            res = [point['xx']] + res + [point['yy']]
            #res.append(point['xx'])
            #res.append(point['yy'])
    return res

'''
Input: a foods object (dict)
Output: a vector which contains significant data about the foods
'''
def process_foods(foods):
    pass

'''
Input: a snake object (it's not the same as snakes object!)
Output: a vector which contains significant data about the agents snake
'''
def process_my_snake(snake):
    pass