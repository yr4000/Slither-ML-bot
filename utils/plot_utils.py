'''
This file will contain all the functions which we will use in order to plot graphic
representations of our experiments results.
'''

import matplotlib
#matplotlib.use('Qt4Agg')    #TODO: not sure it is necessary...
#matplotlib.use('Agg') #in order not to display the plot
import matplotlib.pyplot as plt

def plot_graph(points,title,file_name):
    plt.plot(points)
    plt.title(title)  #TODO: the title should not be the same all the time but dynamic according to some parameters
    #plt.savefig("..\\..\\graphs\\file_name")    #TODO: so do the file name
    plt.savefig(file_name)
