import os
import json

def generate_parameters():
    generate_CNN_params()
    generate_DQN_PARAMS()
    generate_PG_params()
    generate_DQN_CONST_OPT_PARAMS()

def generate_DQN_PARAMS():
    parmeters = {
        "SLICES_NO": 32,
        "VAR_NO": 12,
        "MAX_STEPS": 10000,
        "NUM_OF_EPOCHS": 10,
        "NUM_OF_GAMES_FOR_TEST": 5,
        "LEARN_RATE": 1e-4,

        "FRAMES_PER_OBSERVATION": 4,
        "LAST_RAW_SCORES_SIZE": 200,
        "MEMORY_SIZE": 400000,
        "MIN_MEMORY_SIZE_FOR_TRAINING": 40000,
        "MINI_BATCH_SIZE": 100,
        "FUTURE_REWARD_DISCOUNT": 0.8,
        "INITIAL_RANDOM_ACTION_PROB": 1,
        "FINAL_RANDOM_ACTION_PROB": 0.001,
        "WRITE_TO_LOG": True,
        "WRITE_TO_LOG_EVERY": 100,

        "WEIGHTS_FILE": "DQN_weights.pkl",
        "BEST_WEIGHTS": "DQN_best_weights.pkl",
        "WEIGHTS_DIR": "./weights/",
        "DO_LOAD_WEIGHTS": False,
        "WEIGHTS_TO_LOAD": "./weights/DQN_weights.pkl",

        "LEARN_FROM_EXPERT": False,
        "TEST_MODE": False
    }

    with open('parameters/DQN_Params.json', 'w') as outfile:
        json.dump(parmeters, outfile)

#default CNN parameters
def generate_CNN_params():
    params = {
            "OUTPUT_DIM": 64,
            "INPUT_DIM": 576,
            "FRAMES_PER_OBSERVATION": 4,
            "PLN": 2,
            "NUM_OF_CHANNELS_LAYER1": 1,
            "NUM_OF_CHANNELS_LAYER2": 16,
            "NUM_OF_CHANNELS_LAYER3": 32,
            "SIZE_OF_FULLY_CONNECTED_LAYER_1": 256,
            "SIZE_OF_FULLY_CONNECTED_LAYER_2": 128,
            "SIZE_OF_FULLY_CONNECTED_LAYER_3": 64,
            "KEEP_RATE": 0.9,
            "NUMBER_OF_FC_LAYERS": 3
    }
    with open('parameters/CNN_Params.json', 'w') as outfile:
        json.dump(params, outfile)

#default Policy Gradient parameters
def generate_PG_params():
    params = {
        "SLICES_NO": 32,
        "OUTPUT_DIM": 64,
        "INPUT_DIM": 576,
        "PLN": 2,
        "NUM_OF_CHANNELS_LAYER1": 1,
        "NUM_OF_CHANNELS_LAYER2": 16,
        "NUM_OF_CHANNELS_LAYER3": 32,
        "SIZE_OF_FULLY_CONNECTED_LAYER_1": 256,
        "SIZE_OF_FULLY_CONNECTED_LAYER_2": 128,
        "SIZE_OF_FULLY_CONNECTED_LAYER_3": 64,
        "VAR_NO": 12,
        "KEEP_RATE": 0.9,
        "EPSILON_FOR_EXPLORATION": 0.01,
        "MAX_GAMES": 500,
        "STEPS_UNTIL_BACKPROP": 100,
        "BATCH_SIZE": 10,
        "WEIGHTS_FILE": "PG_weights.pkl",
        "BEST_WEIGHTS": "PG_best_weights.pkl",
        "LOAD_WEIGHTS": False,
        "BEGINING_SCORE": 10,
        "WRITE_TO_LOG": 50
    }

    with open('parameters/Policy_Gradient_Params.json', 'w') as outfile:
        json.dump(params, outfile)

def generate_DQN_CONST_OPT_PARAMS():
    parmeters = {
        "SLICES_NO": 32,
        "VAR_NO": 12,
        "MAX_STEPS": 10000,
        "NUM_OF_EPOCHS": 10,
        "EPISODE_SIZE": 1000,
        "LOCAL_HORIZON": 4,
        "PENALTY_COEFF": 0.6,
        "NUM_OF_GAMES_FOR_TEST": 5,
        "LEARN_RATE": 1e-4,

        "FRAMES_PER_OBSERVATION": 4,
        "LAST_RAW_SCORES_SIZE": 200,
        "MEMORY_SIZE": 400000,
        "MIN_MEMORY_SIZE_FOR_TRAINING": 40000,
        "MINI_BATCH_SIZE": 100,
        "FUTURE_REWARD_DISCOUNT": 0.8,
        "INITIAL_RANDOM_ACTION_PROB": 1,
        "FINAL_RANDOM_ACTION_PROB": 0.001,
        "WRITE_TO_LOG": True,
        "WRITE_TO_LOG_EVERY": 100,

        "WEIGHTS_FILE": "DQN_weights.pkl",
        "BEST_WEIGHTS": "DQN_best_weights.pkl",
        "WEIGHTS_DIR": "./weights/",
        "DO_LOAD_WEIGHTS": False,
        "WEIGHTS_TO_LOAD": "./weights/DQN_weights.pkl",

        "LEARN_FROM_EXPERT": False,
        "TEST_MODE": False
    }

    with open('parameters/DQN_Const_Opt_Params.json', 'w') as outfile:
        json.dump(parmeters, outfile)

#if parameters does not exist - generate them
directory = './parameters/'
if (not os.path.exists(directory)):
    os.makedirs(directory)
    generate_parameters()