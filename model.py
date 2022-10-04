from autograd import grad
from autograd.misc import flatten
import autograd.numpy as np
from core import compute_class_probabilities, loss


class Model:
    # Initialize the parameters to random values.
    np.random.seed(0)

    def __init__(self, input_dim: int) -> None:
        self._model: dict = {}
        self._activation_functions: list = []
        self._input_dim: int = input_dim
        self._n_layers: int = 0

    def add_layer(self, n_neurons: int, activation: callable) -> None:
        if self._n_layers == 0:
            neurons_prev = self._input_dim
        else:
            neurons_prev = self._model[f"b{self._n_layers}"].shape[1]

        _w = np.random.randn(neurons_prev, n_neurons) / np.sqrt(neurons_prev)
        _b = np.zeros((1, n_neurons))
        _func = activation

        self._n_layers += 1
        self._model[f"W{self._n_layers}"] = _w
        self._model[f"b{self._n_layers}"] = _b
        self._activation_functions.append(_func)

    def _compute_class_probabilities(self, inputs: np.ndarray) -> np.ndarray:
        return compute_class_probabilities(
            self._model, self._activation_functions, inputs, self._n_layers)

    # loss function for a 3-layer MLP
    def _loss(self, x_: np.ndarray, y_: np.ndarray) -> float:
        return loss(
            self._model, self._activation_functions, x_, y_, self._n_layers)

    # forward propagation
    def predict(self, x: np.ndarray) -> np.ndarray:
        # Forward propagation to calculate our predictions
        return np.argmax(self._compute_class_probabilities(x), axis=1)

    def train(self, x_: np.ndarray, y_: np.ndarray,
              epochs: int = 1000, print_loss: bool = True,
              epsilon: float = 0.1) -> None:
        assert self._n_layers > 0, \
            "You have to add at least 1 layer to be able to fit the classifier"
        passes: int = x_.shape[0] * epochs
        # Beginning of the gradient descent
        for i in range(0, passes):

            # computing the derivative by Automatic Differentiation
            gradient_loss = grad(lambda model_: loss(
                model_, self._activation_functions, x_, y_, self._n_layers))

            # flattening nested containers containing np arrays
            # Returns 1D np array and an unflatten function.
            model_flat, unflatten_m = flatten(self._model)
            grad_flat, unflatten_g = flatten(gradient_loss(self._model))

            # gradient descendW
            model_flat -= grad_flat * epsilon
            self._model = unflatten_m(model_flat)

            # TODO: implement early stopping
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset,
            # so we don't want to do it too often.
            if print_loss and i % 10000 == 0:
                print("Loss after iteration %i: %f" % (i, self._loss(x_, y_)))
