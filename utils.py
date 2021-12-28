from sympy import Symbol, Matrix, lambdify
import numpy as np

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
    # return x3**2 - x1**2 - x2**2


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
    
    # return 1 - x3**2 - x2**2

def F1_curve(l1, f1_type):
    return (F1(l1[0],l1[1],l1[2], f1_type))**2

def F2_curve(l1, f2_type):
    return (F2(l1[0],l1[1],l1[2],f2_type))**2

def G(l1, f1_type, f2_type):
    return (F1(l1[0],l1[1],l1[2],f1_type))**2 + (F2(l1[0],l1[1],l1[2],f2_type))**2

x1 = Symbol('x')
x2 = Symbol('y')
x3 = Symbol('z')
X = Matrix([x1, x2, x3])

def generate_F1_sample(N, f1_type):
    y_F1 = F1_curve(X, f1_type)
    y_F1_prime1 = y_F1.diff(x1)
    y_F1_prime2 = y_F1.diff(x2)
    y_F1_prime3 = y_F1.diff(x3)
    y_F1_prime_mat = Matrix([y_F1_prime1,y_F1_prime2,y_F1_prime3])
    call_y_F1_prime_mat = lambdify((x1,x2,x3), y_F1_prime_mat, 'numpy')
    call_y_F1 = lambdify((x1, x2,x3, f1_type), y_F1, 'numpy')

    lr = 0.01
    samples = []
    for i in range(N):
        x = np.random.uniform(-2, 2, 3)
        G_value = call_y_F1(x[0], x[1], x[2], f1_type)
        while(G_value > 1.0e-7):
            grad = call_y_F1_prime_mat(x[0], x[1], x[2])
            x = x - (lr*grad).reshape((3,))
            G_value = call_y_F1(x[0], x[1], x[2],f1_type) 
        samples.append(x)   
    return np.array(samples)


def generate_F2_sample(N, f2_type):
    y_F2 = F2_curve(X,f2_type)
    y_F2_prime1 = y_F2.diff(x1)
    y_F2_prime2 = y_F2.diff(x2)
    y_F2_prime3 = y_F2.diff(x3)
    y_F2_prime_mat = Matrix([y_F2_prime1,y_F2_prime2,y_F2_prime3])
    call_y_F2_prime_mat = lambdify((x1, x2,x3), y_F2_prime_mat, 'numpy')
    call_y_F2 = lambdify((x1, x2,x3,f2_type), y_F2, 'numpy')

    lr = 0.01
    samples = []
    for i in range(N):
        x = np.random.uniform(-2, 2, 3)
        G_value = call_y_F2(x[0], x[1], x[2],f2_type)
        while(G_value > 1.0e-7):
            grad = call_y_F2_prime_mat(x[0], x[1], x[2])
            x = x - (lr*grad).reshape((3,))
            G_value = call_y_F2(x[0], x[1], x[2],f2_type)    
        samples.append(x)
    return np.array(samples)



def generate_intersection_sample(N, f1_type, f2_type):

    y = G(X,f1_type, f2_type)

    yprime1 = y.diff(x1)
    yprime2 = y.diff(x2)
    yprime3 = y.diff(x3)
    yprime_mat = Matrix([yprime1,yprime2,yprime3])
    call_yprime_mat = lambdify((x1, x2,x3), yprime_mat, 'numpy')
    call_y = lambdify((x1, x2,x3,f1_type, f2_type), y, 'numpy')

    lr = 0.01
    samples = []
    for i in range(N):
        x = np.random.uniform(-2, 2, 3)
        G_value = call_y(x[0], x[1], x[2],f1_type, f2_type)
        while(G_value > 1.0e-7):
            grad = call_yprime_mat(x[0], x[1], x[2])
            x = x - (lr*grad).reshape((3,))
            G_value = call_y(x[0], x[1], x[2], f1_type, f2_type)  
        samples.append(x)  
    return np.array(samples)


