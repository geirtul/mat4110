def dataset_one():
    import numpy as np
    n = 30
    start = -2
    stop = 2
    x = np.linspace(start, stop, n)
    eps = 1
    np.random.seed(1) # use same seed every time
    r = np.random.random(n) * eps
    y = x*(np.cos(r + 0.5*x**3) + np.sin(0.5*x**3))
    return x,y


def dataset_two():
    import numpy as np
    n = 30
    start = -2
    stop = 2
    x = np.linspace(start, stop, n)
    eps = 1
    np.random.seed(1) # use same seed every time
    r = np.random.random(n) * eps
    y = 4*x**5 - 5*x**4 - 20*x**3 + 10*x*x + 40*x + r
    return x,y
