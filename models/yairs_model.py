import tensorflow as tf
import pickle as pkl

import json


with open('../observation.json') as json_data:
    d = json.load(json_data)
    print(d)

