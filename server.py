from flask import Flask, render_template, request, redirect, Response,jsonify
from flask_cors import CORS, cross_origin
import json
from datetime import datetime
import time as t

app = Flask(__name__)
cors = CORS(app)        #This is needed for the server to be able to send responses

'''
@app.route('/')
def output():
    return render_template('index.html', name='Joe')
    #return redirect("This is a temporary implementation.")
'''

#this route will get the data from the client, send it to the agent and then send the result back to the client.
@app.route('/model',methods = ['POST'])
def ask_model():
    data = request.get_json(force=True)

    print("Request "+str(data['message_id'])+' sent in: '+str(data['hours'])+':'+str(data['minutes'])+':'+str(data['seconds']))

    with open('observation.json', 'w') as outfile:
        json.dump(data, outfile)
    #print("data: " + str(data) + '\n')
    #print("Score: " + str(data['score']) + '\n')
    #print("Foods: "+str(data['foods']) + '\n')
    #print("Preys: " + str(data['preys']) + '\n')
    #print("Snake: "+ str(data['snake']) + '\n')
    #print("input: " + str(data['input']) + '\n')


    # TODO: add sleep? not a good idea since there si already a lag
    #t.sleep(0.05)


    #gets action from file
    try:
        with open('action.json') as json_data:
            res = json.load(json_data)
    except:
        res = {'action': 0,
               'do_accelerate': 0,
               'request_id': -1,
               'commit_sucide': False}        #TODO: change to a better default

    # calculate angle using r and x

    '''
    teta = math.acos(data['x'] / data['r'])
    teta += math.pi / 18
    res['x'] = data['r']*math.cos(teta)
    res['y'] = data['r']*math.sin(teta)
    '''

    #print(res)
    time = datetime.now().time()
    print("Responded to request "+str(data['message_id'])+' in: '+str(time.hour)+':'+str(time.minute)+':'+str(time.second))
    return jsonify(res)



if __name__ == '__main__':
    # run!
    #the "IP:Port" the server listens to is "localhost:5000"
    app.run()
