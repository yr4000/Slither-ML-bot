from flask import Flask, render_template, request, redirect, Response,jsonify
from flask_cors import CORS, cross_origin
import random, json
import math

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
    with open('observation.json', 'w') as outfile:
        json.dump(data, outfile)
    #print("data: " + str(data) + '\n')
    #print("Score: " + str(data['score']) + '\n')
    #print("Foods: "+str(data['foods']) + '\n')
    #print("Preys: " + str(data['preys']) + '\n')
    #print("Snake: "+ str(data['snake']) + '\n')
    #print("input: " + str(data['input']) + '\n')

    print("x: "+str(data['x']) + ", y: " + str(data['y']) + ", r: " + str(data['r']) + "\n")

    #gets action from file
    try:
        with open('action.json') as json_data:
            res = json.load(json_data)
    except:
        res = {}

    # calculate angle using r and x

    '''
    teta = math.acos(data['x'] / data['r'])
    teta += math.pi / 18
    res['x'] = data['r']*math.cos(teta)
    res['y'] = data['r']*math.sin(teta)
    '''

    print(res)
    return jsonify(res)



if __name__ == '__main__':
    # run!
    #the "IP:Port" the server listens to is "localhost:5000"
    app.run()
