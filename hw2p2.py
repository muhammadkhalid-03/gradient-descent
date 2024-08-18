import numpy as np
import math
import matplotlib.pyplot as plt

"""
Function returns the root of a given function using the Newton-Raphson method

Parameter:
- f: function for which the minimizing vector has to be found
- df: gradient vector
- a: initial guess vector
- max_iter=1000: maximum amount of iterations before the function stops
- tol=1e-6: tolerance for when the function should keep running

Returns:
- x_new: approximate root of the given function
- i: number of iterations that have passed

Note: Returns a message saying that the maximum iterations were reached
      if maximum iterations were reached
"""
def newton(f, df, a, solutionVec, max_iter=100, tol=1e-12):
        x = a
        errorList = [] #list of errors
        for i in range(max_iter):
            currError = np.linalg.norm(x - solutionVec)
            x_new = x - (np.linalg.inv(f4Hessian(x))@df(x))
            # print(x_new)
            # currError = np.linalg.norm(x_new - solutionVec)
            print(currError)
            # Check for convergence
            if np.linalg.norm(x_new-x) < tol:
                return x_new, i+1, f(x), errorList  # Return the root and the number of iterations
            errorList.append(currError)
            x = x_new
        print("Maximum iterations were reached")
        return None, None, None, None



def f4(vec):
    return np.cos(vec[0])+np.sin(vec[1])

def df4(vec):
    return np.array([-math.sin(vec[0]), math.cos(vec[1])])

def f4Hessian(vec):
    return np.array([[-math.cos(vec[0]), 0.0], [0, -math.sin(vec[1])]])







if __name__ == "__main__":

    # Question 1:
    x = np.array([3.0, 4.0]) #initial vector guess
    solutionVecNewton = np.array([3.14158889, 4.71236907])
    newtonMinimum_x, newtonIterations, newtonMinimum_y, errors = newton(f4, df4, x, solutionVecNewton)    #stores minimum as vector
    print("\n\nMinimum x vector for Function 4:", newtonMinimum_x)
    print("\n\nMinimum function value for Function 4:", newtonMinimum_y)
    print("\n\nIterations:", newtonIterations)
    
    #Question 2:
    print(errors)
    # Plot consecutive pairs of errors as a line on a log-log scale
    plt.figure(figsize=(8, 6))
    plt.loglog(errors[:-1], errors[1:], color='blue', marker='o', linestyle='-')
    plt.xlabel('log |ϵn|')
    plt.ylabel('log |ϵn+1|')
    plt.title('Convergence behavior of Newton Method')
    plt.grid(True)
    plt.show()

