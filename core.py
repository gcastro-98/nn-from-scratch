import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0., x)


def identity(x: np.ndarray) -> np.ndarray:
    return x


# NEEDED OUTSIDE TO BE DIFFERENTIATED by autograd

def forward_propagation(model: dict, activation_functions: list,
                        inputs: np.ndarray, n_layers: int) -> np.ndarray:
    assert n_layers, "The model needs at least 1 layer!"
    a_prev: np.ndarray = inputs
    # noinspection PyTypeChecker
    a_curr: np.ndarray = None
    for n_ in range(1, n_layers + 1):
        w_curr = model[f'W{n_}']
        b_curr = model[f'b{n_}']
        z_curr = np.dot(a_prev, w_curr) + b_curr
        a_curr = activation_functions[n_ - 1](z_curr)

        a_prev = a_curr
    return a_curr


def compute_class_probabilities(
        model: dict, activation_functions: list,
        inputs: np.ndarray, n_layers: int) -> np.ndarray:
    exp_scores = np.exp(forward_propagation(model, activation_functions,
                                            inputs, n_layers))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs


def loss(model: dict, activation_functions: list,
         x_: np.ndarray, y_: np.ndarray, n_layers: int) -> float:
    num_examples: int = len(x_)
    probs = compute_class_probabilities(
        model, activation_functions, x_, n_layers)

    # Calculating the loss
    correct_logprobs = -np.log(probs[range(num_examples), y_.astype(int)])
    data_loss = np.sum(correct_logprobs)

    return 1. / num_examples * data_loss
