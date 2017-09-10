'''
This file will contain all the functions which we will use in order to plot graphic
representations of our experiments results.
'''

import matplotlib
#matplotlib.use('Qt4Agg')    #This code is not necessary for most of the users
#matplotlib.use('Agg') #in order not to display the plot
import matplotlib.pyplot as plt
import os
import time
#from scipy import misc

def plot_graph(points,title,file_name,xlabel,ylabel):
    directory = './graphs/'
    if (not os.path.exists(directory)):
        os.makedirs(directory)

    plt.plot(points)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    file_name = directory+file_name+ '_' + time.strftime("%d%m%Y-%H%M%S") + '.png'
    plt.savefig(file_name)
    plt.clf()

'''
def obsrv_to_image(observ,files_name):
    observ = misc.imresize(observ,2500)    #X25 bigger
    misc.toimage(observ).save(files_name + '.png')
'''
'''
a = [12.503910613720384, 12.810353757833768, 12.049875650342624, 13.379854196976501, 12.547900113874721, 11.252502269381427, 12.10732014914525, 12.713186562371158, 14.723609183524047, 13.287215052725037, 13.290201651378544, 14.547379239511551, 42.393825022602691, 17.117635393484655, 15.20554495958012, 13.735472769690796, 19.72686737578697, 17.326440699650593, 21.805319609098873, 14.410597962757402]
'''
