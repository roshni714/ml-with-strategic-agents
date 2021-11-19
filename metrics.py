import numpy as np

def bias(emp_gradients, exp_gradients):
    emp_gradients = np.array(emp_gradients)
    exp_gradients = np.array(exp_gradients)

    assert emp_gradients.shape == exp_gradients.shape

    abs_bias = np.mean(emp_gradients - exp_gradients).item()
    return abs_bias

def variance(emp_gradients):
    variance = np.var(emp_gradients).item()
    return variance


