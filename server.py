from flask import Flask, render_template, request, redirect, Response,jsonify
from flask_cors import CORS, cross_origin
import random, json
import math

from utils.process_utils import process_snakes

app = Flask(__name__)
cors = CORS(app)        #This is needed for the server to be able to send responses


@app.route('/')
def output():
    return render_template('index.html', name='Joe')
    #return redirect("This is a temporary implementation.")

#this route will get the data from the client, send it to the agent and then send the result back to the client.
@app.route('/model',methods = ['POST'])
def ask_model():
    data = request.get_json(force=True)

    snakes_points = process_snakes(data['snakes'])
    #print("data: " + str(data) + '\n')
    #print("Score: " + str(data['score']) + '\n')
    print("Snakes points: "+ str(snakes_points) + '\n')
    #print("Foods: "+str(data['foods']) + '\n')
    #print("Preys: " + str(data['preys']) + '\n')
    #print("Snake: "+ str(data['snake']) + '\n')

    print("x: "+str(data['x']) + ", y: " + str(data['y']) + ", r: " + str(data['r']) + "\n")
    #calculate angle using r and x
    teta = math.acos(data['x']/data['r'])
    teta += math.pi/18
    res = {}
    res['x'] = data['r']*math.cos(teta)
    res['y'] = data['r']*math.sin(teta)
    print("Result: x = " + str(res['x']) + ", y = " + str(res['y']) + ", teta = " + str(teta) + "\n")
    return jsonify(res)



if __name__ == '__main__':
    # run!
    #the "IP:Port" the server listens to is "localhost:5000"
    app.run()
