import numpy as np
import matplotlib.pyplot as plt

def showFacialLandmarks(Y, shift_x, opt):
    # draw nose, mouth, jaw landmarks part-by-part

    plt.plot(Y[0:14:2] + shift_x, Y[1:14:2], opt)
    plt.plot(Y[14:24:2] + shift_x, Y[15:24:2], opt)
    plt.plot(Y[24:52:2] + shift_x, Y[25:52:2], opt)
    plt.plot(Y[[24, 50]] + shift_x, Y[[25, 51]], opt)
    plt.plot(Y[52::2] + shift_x, Y[53::2], opt)
    plt.plot(Y[[52, 74]] + shift_x, Y[[53, 75]], opt)

    return plt