from flask import Flask, render_template, request, redirect, Response,jsonify
#from flask_cors import CORS, cross_origin
import random, json

app = Flask(__name__)
#cors = CORS(app)        #right now this doesn't help...


@app.route('/')
def output():
    return render_template('index.html', name='Joe')
    #return redirect("This is a temporary implementation.")

#this route will get the data from the client, send it to the agent and then send the result back to the client.
@app.route('/model',methods = ['POST'])
def ask_model():
    data = request.get_json(force=True)
    print("data recieved: "+str(data))
    return jsonify({'value': "some response"})



if __name__ == '__main__':
    # run!
    #the "IP:Port" the server listens to is "localhost:5000"
    app.run()
