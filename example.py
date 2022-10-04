import matplotlib
import autograd.numpy as np
from plot import decision_boundary
from sklearn.metrics import accuracy_score

from model import Model
from core import relu, identity

#########################################################################

# Display plots inline and change default figure size
matplotlib.rcParams['figure.figsize'] = (6.0, 4.0)

# Set some parameters
nn_input_dim = 2  # input layer dimensionality
nn_output_dim = 2  # output layer dimensionality

np.random.seed(0)

############################################################################


def __two_spirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points, 1) * noise
    return (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
            np.hstack((np.zeros(n_points), np.ones(n_points))))


def exercise3() -> None:
    """
    We are going to design a 6-layer classifier with ReLU
    as activation function and 6 neurons for each of them.
    """
    # Generate a testing dataset
    x_spir, y_spir = __two_spirals(1000)

    # Generate a testing dataset
    xt_spir, yt_spir = __two_spirals(1000)

    model = Model(nn_input_dim)
    model.add_layer(n_neurons=6, activation=relu)
    model.add_layer(n_neurons=6, activation=relu)
    model.add_layer(n_neurons=6, activation=relu)
    model.add_layer(n_neurons=6, activation=relu)
    model.add_layer(n_neurons=6, activation=relu)
    model.add_layer(n_neurons=nn_output_dim, activation=identity)  # output layer
    model.train(x_=x_spir, y_=y_spir, epochs=50)
    # Use the model for inference
    y_hat = model.predict(xt_spir)
    print("\n The accuracy in the test set is: ", accuracy_score(yt_spir, y_hat))

    decision_boundary(
        xt_spir, yt_spir, model.predict, plot_name='exercise3_class')


if __name__ == '__main__':
    exercise3()
