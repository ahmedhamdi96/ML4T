import numpy as np 
import scipy.optimize as spo
import matplotlib.pyplot as plt

def f(x):
    #f(x) = (x-1.5)^2 + 0.5
    return (x - 1.5) ** 2 + 0.5

def main():
    x_guess = 2.0
    min = spo.minimize(f, x_guess, method = 'SLSQP', options = {'disp' : True})
    print("Minima is at: ({}, {})".format(min.x, min.fun))

    x_points = np.linspace(0, 3, 30)
    y_points = f(x_points)
    plt.plot(x_points, y_points)
    plt.plot(min.x, min.fun, 'ro')
    plt.title("Minima of a Function")
    plt.show()


if __name__ == "__main__":
    main()
