import numpy as np
import math

"""
The function gradDescent calculates a solution vector x using 
a provided alpha (step size), an initial guess vector, a function
and the gradent vector for that function

Parameters:
initGuess - initial guess vector
f - input function that takes in a vector
df - gradient vector that takes in a vector
alpha - pre-defined step size
tol=1e-8 - constant, tolerance for when the function stops at a specific
           iteration
max_iter=100 - constant, maximum number of iterations after which the
               function stops working

Returns:

x - The solution vector where the function is minimized
f(x) - The minimum value of the function
i - number of iterations that have passes
"""

def gradDescent(initGuess, f, df, alpha, tol=1e-8, max_iter=100):
    x= initGuess
    prevX = initGuess
    diff = initGuess
    for i in range(max_iter):
        if (np.linalg.norm(diff) < tol):
            return x, f(x), i+1
        x = x - (alpha*df(x))
        diff = x - prevX
        prevX = x
    return x, f(x), i+1

"""
The function updateAlpha() calculates the updated alpha using
a guess vector and pre-defined A and b vectors.

Parameters:
  x - Guess vector of R^3

Returns:
  alpha (float) - the step size at the current iteration
"""
def updateAlpha(guessVec: np.array):

    # define A
    A = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]])

    # define b
    b = np.array([[2],[0],[2]])

    # Calculate gradient vector
    gradVec = df4(guessVec)

    # Calculate alpha
    alpha = ((gradVec.T)@gradVec) / (((gradVec.T)@A)@gradVec)
    return alpha

"""
The function gradDescent2 calculates a solution vector x using 
a provided alpha (step size) after which the function calculates
its own optimal alpha at each iteration using a 3x3 matrix
and a vector in R^3. It also uses an initial guess vector, 
a function and the gradent vector for that function

Parameters:
initGuess - initial guess vector
f - input function that takes in a vector
df - gradient vector that takes in a vector
alpha - pre-defined step size
tol=1e-8 - constant, tolerance for when the function stops at a specific
           iteration
max_iter=100 - constant, maximum number of iterations after which the
               function stops working

Returns:

x - The solution vector where the function is minimized
f(x) (float) - The minimum value of the function
i - number of iterations that have passed
"""
def gradDescent2(initGuess, f, df, alpha, tol=1e-8, max_iter=100):
    x= initGuess
    prevX = initGuess
    diff = initGuess
    for i in range(max_iter):
        if (np.linalg.norm(diff) < tol):
            return x, f(x), i

        x = x - (alpha*df(x))
        alpha = updateAlpha(x) # update alpha here
        diff = x - prevX
        prevX = x
    return x, f(x), i

"""
The function gradDescent3 calculates a solution vector x using 
a provided alpha (step size) after which the function decreases
alpha by a contraction factor to use in the next guess. It also 
uses an initial guess vector, a function and the gradent vector 
for that function.


Parameters:
initGuess - initial guess vector
f - input function that takes in a vector
df - gradient vector that takes in a vector
alpha - pre-defined step size
contractionFactor - factor by which alpha decreases or increases
tol=1e-8 - constant, tolerance for when the function stops at a specific
           iteration
max_iter=100 - constant, maximum number of iterations after which the
               function stops working

Returns:

x - The solution vector where the function is minimized
f(x) - The minimum value of the function
i - The number of iterations that have passes
"""

def gradDescent3(initGuess, f, df, alpha, contractionFactor, c, tol=1e-8, max_iter=100):
    x= initGuess
    prevX = initGuess
    diff = initGuess
    for i in range(max_iter):
        if (np.linalg.norm(diff) < tol):
            return x, f(x), i
        f_parameter = x+(alpha*(-df(x)))
        while (f(f_parameter) > f(x)+c*alpha*(df(x).T)*(-df(x))).all():
            alpha = contractionFactor * alpha
            f_parameter = x+(alpha*(-df(x)))
        x = x - (alpha*df(x))
        diff = x - prevX
        prevX = x
    return x, f(x), i


def f1(vec):
    return vec[0]**2+vec[1]**2

def df1(vec):
    return 2*vec

def f2(vec):
    return 1e06 * (vec[0]**2) + (vec[1]**2)

def df2(vec):
    return 2e06*vec[0] + 2*vec[1]

def f3(vec):
    return (vec[0]**2)+(vec[1]**2)+(vec[2]**2)+(vec[3]**2)+(vec[4]**2)+(vec[5]**2)

def df3(vec):
    return 2*vec[0]+2*vec[1]+2*vec[2]+2*vec[3]+2*vec[4]+2*vec[5] 

def f4(vec:np.array):
    A = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]])
    b = np.array([[2],[0],[2]])
    return 0.5 * (vec.T)@A@vec - (b.T)@vec

def df4(vec:np.array):
    A = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]])
    b = np.array([[2],[0],[2]])
    return A@vec-b

if __name__ == "__main__":


    # When using 5.0 for the guess vector and alpha=0.1, in 95 iterations i got:
    # Minimum x vector: [3.10827023e-09 3.10827023e-09]
    # Minimum function value: 1.9322687615086322e-17

    # When using 5.0 for the guess vector and alpha=1.0, in 100 iterations I got:
    # Minimum x vector: [5. 5.]
    # Minimum function value: 50.0
    # which means that I overstepped by too large of an amount

    # With an alpha of 0.05 and guess vector of 5.0, in 100 iterations I got:
    # Minimum x vector: [0.00013281 0.00013281]
    # Minimum function value: 3.527539554327655e-08
    # which means that it converged slower than the other cases and thus 
    # got a larger  value compared to the first case

    # The previous ones were stopping later and gave a smaller result
    # All the previous ones were squaring each value in the gradient vector and
    # adding them and then square rooting the result to compare it to the tolerance
    # Now i'm calculating the difference between two consecutive vector guesses
    # and then finding the norm of that difference using the numpy.linalg.norm function
    # and getting the following results:
    # iterations: 86
    # Minimum x vector: [2.31584178e-08 2.31584178e-08]
    # Minimum function value: 1.072624634395409e-15

    #Question 1:
    x = np.array([5.0, 5.0]) #initial vector guess
    alpha = 0.1 #step length
    minimum_x, minimum_y, i = gradDescent(x, f1, df1, alpha)    #stores minimum as vector
    print("\n\nMinimum x vector for Function 1:", minimum_x)
    print("\n\nMinimum function value for Function 1:", minimum_y)
    print("\n\nNumber of iterations:", i)

    alpha = 0.1 #step length
    minimum_x, minimum_y, i = gradDescent(x, f2, df2, alpha)    #stores minimum as vector
    print("\n\nMinimum x vector for Function 2:", minimum_x)
    print("\n\nMinimum function value for Function 2:", minimum_y)
    print("\n\nNumber of iterations:", i)

    x = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0]) #initial vector guess
    alpha = 0.1 #step length
    minimum_x, minimum_y, i = gradDescent(x, f3, df3, alpha)    #stores minimum as vector
    print("\n\nMinimum x vector for Function 3:", minimum_x)
    print("\n\nMinimum function value for Function 3:", minimum_y)
    print("\n\nNumber of iterations:", i)

    # Question 2: (use gradDescent 2 function)
    x = np.array([[10.0], [10.0], [10.0]]) #initial vector guess
    alpha = 0.1 #step length
    minimum_x, minimum_y, i = gradDescent2(x, f4, df4, alpha)    #stores minimum as vector
    print("\n\nMinimum x vector for updated alpha Function:", minimum_x)
    print("\n\nMinimum function value for updated alpha function:", minimum_y)
    print("\n\nNumber of iterations:", i)

    # Question 3: (use gradDescent3 function)
    #   - When I tried a step size of 5.0 for the first question,
    #     it diverged to a large number but when I tried a step
    #     size of 5.0 for the third question, it converged to 0.
    x = np.array([5.0, 5.0]) #initial vector guess
    alpha = 0.1 #step length
    minimum_x, minimum_y, i = gradDescent3(x, f1, df1, alpha, 0.9, 0.1)    #stores minimum as vector
    print("\n\nMinimum x vector for Function 1:", minimum_x)
    print("\n\nMinimum function value for Function 1:", minimum_y)
    print("\n\nNumber of iterations:", i)

    alpha = 0.1 #step length
    minimum_x, minimum_y, i = gradDescent3(x, f2, df2, alpha, 0.9, 0.1)    #stores minimum as vector
    print("\n\nMinimum x vector for Function 2:", minimum_x)
    print("\n\nMinimum function value for Function 2:", minimum_y)
    print("\n\nNumber of iterations:", i)

    x = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0]) #initial vector guess
    alpha = 0.1 #step length
    minimum_x, minimum_y, i = gradDescent3(x, f3, df3, alpha, 0.9, 0.1)    #stores minimum as vector
    print("\n\nMinimum x vector for Function 3:", minimum_x)
    print("\n\nMinimum function value for Function 3:", minimum_y)
    print("\n\nNumber of iterations:", i)