# the task is to solve differential equations via using third and forth-order schemes
import numpy as np
import matplotlib.pyplot as plt


def f1(t, u):
    return (2*t**3 + t**2 - u**2)/(2*t**2*u)
yo1 = 1
x_left1, x_right1 = 1, 2


def f2(t, u):
    return (1 - t*u**2)/(t**2*u - 1)
yo2 = 0
x_left2, x_right2 = 0, 1


def f3(t, u):
    return (u - t**2*u)/t
yo3 = 2
x_left3, x_right3 = 1, 2


def f4(t, u):
    return ((u*t + 1)*u - u*t - u**2)/(t*(2*t-1))
yo4 = 2
x_left4, x_right4 = 1, 2

def f_test(t, u):
    return np.cos(t)

def forth_order_scheme(f, x_left, x_right, Yo, grid_size):
    # Runge classical scheme. returns raw with 10 values
    h = (x_right - x_left)/grid_size
    X = np.linspace(x_left, x_right, grid_size)
    Y = list()
    Y.append(Yo)
    for x in X:
        k1 = h*f(x, Y[-1])
        k2 = h*f(x + h/2, Y[-1] + k1/2)
        k3 = h*f(x + h/2, Y[-1] + k2/2)
        k4 = h*f(x + h, Y[-1] + k3)
        Y.append(Y[-1] + (k1 + 2*k2 + 2*k3 + k4)/6)
    return X, Y

'''
ret_raw = list()
    for i in range(11)[1:]:
        ret_raw.append(Y[grid_size/10*i])
    return ret_raw'''
def third_order_scheme(f, x_left, x_right, Yo, grid_size):
    # Hoit metod, returns raw with 10 values
    h = (x_right - x_left)/grid_size
    X = np.linspace(x_left, x_right, grid_size)
    Y = list()
    Y.append(Yo)
    for x in X:
        k1 = h*f(x, Y[-1])
        k2 = h*f(x + h/3, Y[-1] + k1/3)
        k3 = h*f(x + h*2.0/3, Y[-1] + k2*2.0/3)
        Y.append(Y[-1] + (k1 + 3*k3)/4)
    return X, Y


grid_sizes = [10.0, 20.0, 40.0, 80.0, 160.0]
for i in range(len(grid_sizes)):
    X, Y = forth_order_scheme(f4, x_left4, x_right4, yo4, grid_sizes[i])
    plt.plot(X, Y[:-1])
    X, Y = third_order_scheme(f4, x_left4, x_right4, yo4, grid_sizes[i])
    plt.plot(X, Y[:-1])
plt.show()