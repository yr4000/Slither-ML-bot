'''
This file will contain all the functions which we will use in order to plot graphic
representations of our experiments results.
'''

import matplotlib
#matplotlib.use('Qt4Agg')    #TODO: not sure it is necessary...
#matplotlib.use('Agg') #in order not to display the plot
import matplotlib.pyplot as plt
import os
import time

def plot_graph(points,title,file_name):
    directory = './graphs/'
    if (not os.path.exists(directory)):
        os.makedirs(directory)

    plt.plot(points)
    plt.title(title)  #TODO: the title should not be the same all the time but dynamic according to some parameters
    file_name = directory+file_name+ '_' + time.strftime("%d%m%Y-%H%M%S") + '.png'
    plt.savefig(file_name)
    plt.clf()
