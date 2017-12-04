import matplotlib.pyplot as plt

from core import *


def heatmap(u):
    plt.imshow(u.reshape(int((3 / delta) + 1), -1), cmap='hot')
    plt.colorbar()
    plt.show()
