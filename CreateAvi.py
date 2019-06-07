import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from showFacialLandmarks import showFacialLandmarks

def CreateAvi(Y_pred, Y_groundtruth, filename):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    def update(i):
        print('Saving {}/{}'.format(i, Y_pred.shape[0]))
        plt.cla()
        showFacialLandmarks(Y_groundtruth[i], 0, 'r-')
        showFacialLandmarks(Y_pred[i], 1.5, 'b-')
        plt.axis('equal')
        plt.axis([-1, 2.5, -1.5, 0.5])

    a = anim.FuncAnimation(fig, update, frames=Y_pred.shape[0], repeat=False)
    Writer = anim.writers['ffmpeg']
    writer = Writer(fps=100, metadata=dict(artist='Me'), bitrate=1800)
    a.save(filename=filename, writer=writer)