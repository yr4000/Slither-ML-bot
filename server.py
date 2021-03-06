from flask import Flask, render_template, request, redirect, Response, jsonify
from flask_cors import CORS, cross_origin
import json
from datetime import datetime

app = Flask(__name__)
cors = CORS(app)#This is needed for the server to be able to send responses

#this route will get the data from the client, send it to the agent and then send the result back to the client.
@app.route('/model', methods = ['POST'])
def ask_model():
    data = request.get_json(force=True)
    print("Request "+str(data['message_id'])+' sent in: '+str(data['hours'])+':'+str(data['minutes'])+':'+str(data['seconds']))

    #saves observation to file
    with open('observation.json', 'w') as outfile:
        json.dump(data, outfile)

    #Gets action from file
    try:
        with open('action.json') as json_data:
            res = json.load(json_data)
    except:
        res = {'action': 0,
               'do_accelerate': 0,
               'request_id': -1,
               'commit_sucide': False}

    #print(res)
    time = datetime.now().time()
    print("Responded to request "+str(data['message_id'])+' in: '+str(time.hour)+':'+str(time.minute)+':'+str(time.second))
    return jsonify(res)



if __name__ == '__main__':
    # run!
    #the "IP:Port" the server listens to is "localhost:5000"
    app.run()