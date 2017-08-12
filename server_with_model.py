from flask import Flask, render_template, request, redirect, Response,jsonify
from flask_cors import CORS, cross_origin
import random, json
import math
import model
import tensorflow as tf

OUTPUT_DIM = 8
#batch_size = 128
INPUT_DIM = 400
MAX_GAMES = 20
BATCH_SIZE = 1

# TODO: what is dropout?!
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

# build the computational graph
states = tf.placeholder(tf.int8, [None, INPUT_DIM], name='states')
actions = tf.placeholder(tf.int8, [None, OUTPUT_DIM], name='chosen_actions')
accu_rewards = tf.placeholder(tf.float32, [None, OUTPUT_DIM], name='accumulated_rewards')


app = Flask(__name__)
cors = CORS(app)        #This is needed for the server to be able to send responses

x = tf.placeholder('float', [None, INPUT_DIM])
y = tf.placeholder('float')

def main():
    prediction = model.convolutional_neural_network(x)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        states_history, action_history, reward_history = [], [], []
        for game_counter in range(MAX_GAMES):
            #init_fake_state = tf.zeros([INPUT_DIM], dtype=tf.int8)
            #TODO: start a new game

            while snake is not None:# we have to recieve the data before using it! and check the syntax!
                #TODO: get state, feed it to the network and send the action to the bot
                #TODO: save the reward,state,action
            if (game_counter % BATCH_SIZE) == 0:
                #TODO: calc_gradients_and_update()
                states_history, action_history, reward_history = [], [], []

    app.run()


@app.route('/')
def output():
    return render_template('index.html', name='Joe')
    #return redirect("This is a temporary implementation.")

#this route will get the data from the client, send it to the agent and then send the result back to the client.
@app.route('/model',methods = ['POST'])
def ask_model():
    data = request.get_json(force=True)

    #print("data: " + str(data) + '\n')
    #print("Score: " + str(data['score']) + '\n')
    #print("Foods: "+str(data['foods']) + '\n')
    #print("Preys: " + str(data['preys']) + '\n')
    #print("Snake: "+ str(data['snake']) + '\n')
    #import pdb; pdb.set_trace()
    print("input: " + str(data['input']) + '\n')

    model.apply_neural_network(data['input'], data['score'], data['snake'])

'''
    print("x: "+str(data['x']) + ", y: " + str(data['y']) + ", r: " + str(data['r']) + "\n")
    #calculate angle using r and x
    teta = math.acos(data['x']/data['r'])
    teta += math.pi/18
    res = {}
    res['x'] = data['r']*math.cos(teta)
    res['y'] = data['r']*math.sin(teta)
    print("Result: x = " + str(res['x']) + ", y = " + str(res['y']) + ", teta = " + str(teta) + "\n")
    return jsonify(res)
'''


if __name__ == '__main__':
    # run!
    #the "IP:Port" the server listens to is "localhost:5000"
    main()
