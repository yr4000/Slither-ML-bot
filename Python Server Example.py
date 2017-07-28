#import socketserver

#!flask/bin/python

import sys

from flask import Flask, render_template, request, redirect, Response,jsonify
import random, json

app = Flask(__name__)


@app.route('/')
def output():
    # serve index template
    return render_template('index.html', name='Joe')
    #return redirect("This text will appear on the screen")


@app.route('/receiver', methods = ['POST'])
def worker():
    print("execute worker()")
    # read json + reply
    data = request.get_json(force=True)
    print("data = "+str(data))
    if data is None:
        return "Recieved data as None"

    result = int(data['value'])

    #this is the example from the original guide:
    #http://www.makeuseof.com/tag/python-javascript-communicate-json/
    '''
    result = ''

    for item in data:
    # loop over every row
    make = str(item['make'])
    if (make == 'Porsche'):
        result += make + ' -- That is a good manufacturer\n'
    else:
        result += make + ' -- That is only an average manufacturer\n'
    '''

    #returns a response in a json format.
    return jsonify({"result":result+1})

if __name__ == '__main__':
    # run!
    #app.run("0.0.0.0", "5010")
    app.run()
