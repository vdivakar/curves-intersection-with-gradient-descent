from sympy import Symbol, Matrix, lambdify
import numpy as np
from sympy.tensor.array import derive_by_array

# defining sympy variables for equations
x1 = Symbol('x')
x2 = Symbol('y')
x3 = Symbol('z')
X = Matrix([x1, x2, x3])

# equation of Fxn-1 depending upon the input
def F1(x1, x2, x3, f1_type):
    x = np.array([x1, x2, x3])
    if f1_type == "DoubleCone1":
        return x3**2 - x1**2 - x2**2
    elif f1_type == "Cylinder1":
        return 1 - x3**2 - x2**2
    elif f1_type == "Sphere1":
        a = np.array([0.36, 0.56, 0.46])
        R = 0.5    
        x_a = x-a
        x_a = np.sum((x_a)**2)
        return x_a - R**2

# equation of Fxn-2 depending upon the input
def F2(x1, x2, x3, f2_type):
    x = np.array([x1, x2, x3])
    if f2_type == "DoubleCone2":
        return (x2-0.90)**2 - x1**2 - x3**2
    elif f2_type == "Cylinder2":
        return 1 - x1**2 - x2**2
    elif f2_type == "Sphere2":
        a = np.array([0.90, 0.33, 0.55])
        R = 0.5    
        x_a = x-a
        x_a = np.sum((x_a)**2)
        return x_a - R**2

# Graph of input functions is F(x) = 0
# Minimise (F(x))^2 to 0 using gradient descent
def F1_curve(l1, f1_type):
    return (F1(l1[0],l1[1],l1[2], f1_type))**2

def F2_curve(l1, f2_type):
    return (F2(l1[0],l1[1],l1[2],f2_type))**2

# Graph of intersection is G(x) = (F1(x))^2 + (F2(x))^2
def G(l1, f1_type, f2_type):
    return (F1(l1[0],l1[1],l1[2],f1_type))**2 + (F2(l1[0],l1[1],l1[2],f2_type))**2

# Generate N points of fxn-1
# To speed up: Instead of using a fixed learning-rate, 
# I've used optimised value for each step.
# It involves Hessian (double derivate of function).
# Separate post for it later. You can use fixed lr too.
def generate_F1_sample(N, f1_type):
    y_F1 = F1_curve(X, f1_type)

    first_derivative = derive_by_array(y_F1, (x1, x2, x3))
    hessian = derive_by_array(first_derivative, (x1, x2, x3))

    #print(len(list(Matrix(hessian).eigenvals().keys())))
    #for x in list(Matrix(hessian).eigenvals().keys()):
    #    print(x)

    numerator = Matrix(first_derivative).dot(first_derivative)
    denominator = Matrix(first_derivative).T * Matrix(hessian) * Matrix(first_derivative)
    call_numerator = lambdify((x1,x2,x3), numerator, 'numpy')
    call_denominator = lambdify((x1,x2,x3), denominator, 'numpy')

    y_F1_prime_mat = Matrix(first_derivative)

    call_y_F1_prime_mat = lambdify((x1,x2,x3), y_F1_prime_mat, 'numpy')
    call_y_F1 = lambdify((x1, x2,x3, f1_type), y_F1, 'numpy')

    samples = []
    for _ in range(N):
        # Generate a random point
        x = np.random.uniform(-2, 2, 3)
        F1_value = call_y_F1(x[0], x[1], x[2], f1_type)
        # Repeate until value of fxn is ~ zero
        while(F1_value > 1.0e-7):
            grad = call_y_F1_prime_mat(x[0], x[1], x[2])
            lr = call_numerator(x[0], x[1], x[2]) / call_denominator(x[0], x[1], x[2])
            # update x using gradient descent
            x = x - (lr*grad).reshape((3,))
            F1_value = call_y_F1(x[0], x[1], x[2],f1_type) 
        samples.append(x)   
    return np.array(samples)

# Similar as above function
def generate_F2_sample(N, f2_type):
    y_F2 = F2_curve(X,f2_type)

    first_derivative = derive_by_array(y_F2, (x1, x2, x3))
    hessian = derive_by_array(first_derivative, (x1, x2, x3))

    numerator = Matrix(first_derivative).dot(first_derivative)
    denominator = Matrix(first_derivative).T * Matrix(hessian) * Matrix(first_derivative)
    call_numerator = lambdify((x1,x2,x3), numerator, 'numpy')
    call_denominator = lambdify((x1,x2,x3), denominator, 'numpy')

    y_F2_prime_mat = Matrix(first_derivative)

    call_y_F2_prime_mat = lambdify((x1, x2,x3), y_F2_prime_mat, 'numpy')
    call_y_F2 = lambdify((x1, x2,x3,f2_type), y_F2, 'numpy')

    samples = []
    for _ in range(N):
        x = np.random.uniform(-2, 2, 3)
        F2_value = call_y_F2(x[0], x[1], x[2],f2_type)
        while(F2_value > 1.0e-7):
            grad = call_y_F2_prime_mat(x[0], x[1], x[2])
            lr = call_numerator(x[0], x[1], x[2]) / call_denominator(x[0], x[1], x[2])
            x = x - (lr*grad).reshape((3,))
            F2_value = call_y_F2(x[0], x[1], x[2],f2_type)    
        samples.append(x)
    return np.array(samples)

# Using vanilla gradient descent.
# [To Do] Optimised alpha not giving proper convergence
def generate_intersection_sample(N, f1_type, f2_type):

    y = G(X,f1_type, f2_type)

    first_derivative = derive_by_array(y, (x1, x2, x3))
    hessian = derive_by_array(first_derivative, (x1, x2, x3))

    # print(list(Matrix(hessian).eigenvals().keys()))
    # print("\n")
    # print(len(list(Matrix(hessian).eigenvals().keys())))
    # for x in list(Matrix(hessian).eigenvals().keys()):
    #     print(x)

    numerator = Matrix(first_derivative).dot(first_derivative)
    denominator = Matrix(first_derivative).T * Matrix(hessian) * Matrix(first_derivative)
    call_numerator = lambdify((x1,x2,x3), numerator, 'numpy')
    call_denominator = lambdify((x1,x2,x3), denominator, 'numpy')

    yprime_mat = Matrix(first_derivative)

    call_yprime_mat = lambdify((x1, x2,x3), yprime_mat, 'numpy')
    call_y = lambdify((x1, x2,x3,f1_type, f2_type), y, 'numpy')

    samples = []
    for _ in range(N):
        x = np.random.uniform(-2, 2, 3)
        G_value = call_y(x[0], x[1], x[2],f1_type, f2_type)
        while(G_value > 1.0e-7):
            grad = call_yprime_mat(x[0], x[1], x[2])
            # lr = call_numerator(x[0], x[1], x[2]) / call_denominator(x[0], x[1], x[2])
            # print(lr)
            lr = 0.02 # Using Fixed alpha here
            x = x - (lr*grad).reshape((3,))
            G_value = call_y(x[0], x[1], x[2], f1_type, f2_type)  
        samples.append(x)  
    return np.array(samples)


