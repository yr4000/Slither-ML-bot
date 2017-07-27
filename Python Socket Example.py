#import socketserver

#!flask/bin/python

import sys

from flask import Flask, render_template, request, redirect, Response
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
    result = ''

    for item in data:
        # loop over every row
        make = str(item['make'])
        if (make == 'Porsche'):
            result += make + ' -- That is a good manufacturer\n'
        else:
            result += make + ' -- That is only an average manufacturer\n'

    return result

if __name__ == '__main__':
    # run!
    #app.run("0.0.0.0", "5010")
    app.run()

'''
class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The RequestHandler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def handle(self):
        # self.request is the TCP socket connected to the client
        self.data = self.request.recv(1024).strip()
        print("{} wrote:".format(self.client_address[0]))
        print(self.data)
        # just send back the same data, but upper-cased
        self.request.sendall(self.data.upper())

if __name__ == "__main__":
    print("Started socket")
    HOST, PORT = "localhost", 9999

    # Create the server, binding to localhost on port 9999
    server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()
    print("finish socket") #will not run this code ever
'''

