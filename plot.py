import os.path

import matplotlib.pyplot as plt
import numpy as np


def decision_boundary(x_: np.ndarray, y_: np.ndarray,
                      pred_func: callable,
                      plot_name: str = 'exercise3') -> None:
    # Set min and max values and give it some padding
    x_min, x_max = x_[:, 0].min() - .5, x_[:, 0].max() + .5
    y_min, y_max = x_[:, 1].min() - .5, x_[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, z, alpha=0.45)
    plt.scatter(x_[:, 0], x_[:, 1], c=y_, alpha=0.45)
    plt.title("Decision boundary plot for exercise 3")
    plt.show()
    if not os.path.isdir('output'):
        os.mkdir('output')
    plt.savefig(f'output/{plot_name}.png')
