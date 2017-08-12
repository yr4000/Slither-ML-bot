'''
This file will contain all the functions that process the incformation
in order to turn it to a vector which will be the input of the model

NOTE: since it makes the snakes slower to do the input process on the server side,
      we decided to do it on the client side (at bot.user.js)
'''

import numpy as np

'''
Input: a snakes object (dict)
Output: a vector which contains significant data about the snakes
'''
def process_snakes(snakes):
    pass

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