import math
import imageio
import numpy as np
import matplotlib.pylab as plt
import cv2
from scipy.io import loadmat

from pybpl.util import dist_along_traj
from pybpl.matlab.bottomup import generate_random_parses



def plot_stroke(ax, stk, color, lw=2):
    if len(stk) > 1 and dist_along_traj(stk) > 0.01:
        ax.plot(stk[:,0], -stk[:,1], color=color, linewidth=lw)
    else:
        ax.plot(stk[0,0], -stk[0,1], color=color, linewidth=lw, marker='.')

def plot_parse(ax, strokes, lw=2):
    ns = len(strokes)
    print ("this is the number of strokes %d" %ns)
    
    colors = ['r','g','b','m','c', 'y', 'k', 'r', 'g']
    for i in range(ns):
        #print ("this is a list of strokes" ,strokes[i])
       
        plot_stroke(ax, strokes[i], colors[i], lw)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0,105)
    ax.set_ylim(105,0)

def main():
    # load image to numpy binary array
    #img = imageio.imread('./numpy_binarization.jpg')
    img = cv2.imread("./final6.png", 0) #reads binary image after preprocessing
    img = np.array(img > 200)
   
    print("loading image done")

    # generate random parses
    #the output is S_Walks 
    #parses[i]= list of the strokes for each of the random walks
    #parses[i][j]= representation of each stroke (don't know what)
    parses = generate_random_parses(img, seed=3)
    print ("This is the S_walks", parses[0])
    
    print("generate parses done")

    # plot parsing results
    nparse = len(parses)
    print("this is the number of parses %d" %nparse)
    
    n = math.ceil(nparse/50) #initially /10
    print(n)
    
    m = 10 #initially 10
    fig, axes = plt.subplots(n,m+1,figsize=(m+1, n))
    
    # first column - input column 
    axes[0,0].imshow(img, cmap=plt.cm.binary)
    axes[0,0].set_xticks([]); axes[0,0].set_yticks([])
    axes[0,0].set_title('Input')
    
    for i in range(1,n):
        axes[i,0].set_axis_off()
    
    # remaining_columns
    for i in range(n): #n de linhas 
        for j in range(1,m+1): #n colunas
            ix = i*m + (j-1) #only getting some of the random walks from the S_walk list
            
           
            if ix >= nparse:
                axes[i,j].set_axis_off()
                continue
            
            plot_parse(axes[i,j], parses[ix]) #parses[ix] gets the list o strokes from random walk ix called "strokes"
    plt.subplots_adjust(hspace=0., wspace=0.)
    plt.show()
    
    #indx = loadmat('indexes.mat')
    #sub_ids = np.asarray(indx, dtype=object)
    #print (sub_ids)


if __name__ == '__main__':
    main()