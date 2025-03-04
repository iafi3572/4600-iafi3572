import numpy as np
import matplotlib.pyplot as plt


def driver():
    f = lambda x: 1 / (1 + (10 * x) ** 2)
    a = -1
    b = 1

    # Create points to evaluate
    Neval = 1000
    xeval = np.linspace(a, b, Neval)

    # Number of intervals
    Nint = 10

    # Evaluate the linear spline
    yeval = eval_lin_spline(xeval, Neval, a, b, f, Nint)

    # Evaluate f at the evaluation points
    fex = f(xeval)

    # Plot the results
    plt.plot(xeval, yeval, label="Linear Spline")
    plt.plot(xeval, fex, label="Exact Function")
    plt.plot(xeval, abs(yeval-fex), label="Error")
    plt.legend()
    plt.savefig("plot1.png")


def subroutine(x0, x1, f, alpha):
    y0 = f(x0)
    y1 = f(x1)
    m = (y1 - y0) / (x1 - x0)
    return m * (alpha - x0) + y0


def eval_lin_spline(xeval, Neval, a, b, f, Nint):
    # Create the intervals for piecewise approximations
    xint = np.linspace(a, b, Nint + 1)

    # Create vector to store the evaluation of the linear splines
    yeval = np.zeros(Neval)

    for jint in range(Nint):
        # Define the interval [xint[jint], xint[jint+1]]
        x0 = xint[jint]
        x1 = xint[jint + 1]

        # Find indices of xeval in the interval
        indices = np.where((xeval >= x0) & (xeval <= x1))[0]

        # Evaluate the spline for the points in this interval
        for idx in indices:
            yeval[idx] = subroutine(x0, x1, f, xeval[idx])

    return yeval


# Run the driver function
driver()
