# Slither-ML-bot

In this project we attempted to create a machine learning model to learn the slither.io game. We explored and implemented differented technics which you can inspect in this code.

For further information, it is recommended to read the mini-article and the bot manual.

## Table of Contents
- Prerequisites and Installation
- Usage
- Machine Learning Modes
- Possible Bug

## Prerequisites and Installation
- python (python 3.5 is recommended)
- The following python packages: Tensorflow, Flask, Flask-CORS, Numpy, Matplotlib, Scipy (the last three are included in the Anaconda package manager)

To install the bot please follow the "README" of the original bot: https://github.com/ErmiyaEskandary/Slither.io-bot,
Then please follow our bot manual to install our javascript code.

## Usage
To run the bot execute the following programs in the order presented:
- Execute the proxy server. you can do so through the cmd, by typing "python server.py"
- Start Slither.io and choose your desired machine learning mode.
- Execute a machine learning model to your choice (the DQN model is recommended).

## Machine Learning Modes
There are three possible machine learning modes implemented in this project. you can switch between them by pressing 'M'.
- <u>Disabled</u> - There is no machine learning, the original heuristic bot is in control. 
- <u>ML server mode</u> - The model you run controls the bot.
- <u>IL mode</u> - IL stands for "imitation learning". The heuristic bot is in control, but observations and chosen actions are being sent to your model. The idea is to allow your model to learn from an expert.
- <u>JS ML mode</u> - This was an attempt to implement our model in javascript in order to overcome the need for server-client communication. This attempt didn't work out, but we left it for demonstration.

### Possible bug
The function getMyScore extracts an information from a div element, which in most browser is the 17th child of the body element. If that's not the case on your local browser, you will see when you run the DQN model that the average score per epoch is always 10. simply switch the index from 17 to 16, or use the developer tools to debug it.

Have fun!
