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

'''
def main():

    a = [77.592619047619053, 56.003, 37.111249999999998, 24.974250000000001, 33.333013513513514, 27.578368421052637,
     23.183499999999999, 44.660999999999987, 21.352898024750122, 25.579499999999999, 19.521507062146892,
     28.124642156862741, 38.173099604823186, 21.128499999999999, 21.236499999999999, 31.360249999999997,
     21.351999999999997, 25.872000000000003, 33.888416666666672, 132.47800000000001, 151.59022234392111]

    plot_graph( a , "expert prefoemance over epochs" , "expert prefoemance over epochs")

if __name__ == "__main__":
    main()

'''
