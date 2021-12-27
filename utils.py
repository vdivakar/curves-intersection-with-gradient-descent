from sympy import Symbol, Matrix, lambdify
import numpy as np

def F1(x1, x2, x3):
    x = np.array([x1, x2, x3])
    a = np.array([0.361522, 0.56793, 0.466301])
    R = 0.5    
    x_a = x-a
    x_a = np.sum((x_a)**2)
    # return x_a - R**2
    return x3**2 - x1**2 - x2**2


def F2(x1, x2, x3):
    x = np.array([x1, x2, x3])
    a = np.array([0.902117, 0.33316, 0.556237])
    R = 0.5    
    x_a = x-a
    x_a = np.sum((x_a)**2)
    # return x_a - R**2
    return 1 - x3**2 - x2**2

def F1_curve(l1):
    return (F1(l1[0],l1[1],l1[2]))**2

def F2_curve(l1):
    return (F2(l1[0],l1[1],l1[2]))**2

def G(l1):
    return (F1(l1[0],l1[1],l1[2]))**2 + (F2(l1[0],l1[1],l1[2]))**2

x1 = Symbol('x')
x2 = Symbol('y')
x3 = Symbol('z')
X = Matrix([x1, x2, x3])
y = G(X)

yprime1 = y.diff(x1)
yprime2 = y.diff(x2)
yprime3 = y.diff(x3)
yprime_mat = Matrix([yprime1,yprime2,yprime3])
call_yprime_mat = lambdify((x1, x2,x3), yprime_mat, 'numpy')
call_y = lambdify((x1, x2,x3), y, 'numpy')


y_F1 = F1_curve(X)
y_F1_prime1 = y_F1.diff(x1)
y_F1_prime2 = y_F1.diff(x2)
y_F1_prime3 = y_F1.diff(x3)
y_F1_prime_mat = Matrix([y_F1_prime1,y_F1_prime2,y_F1_prime3])
call_y_F1_prime_mat = lambdify((x1, x2,x3), y_F1_prime_mat, 'numpy')
call_y_F1 = lambdify((x1, x2,x3), y_F1, 'numpy')


def generate_F1_sample():
    lr = 0.01
    # x = np.random.random(3)
    x = np.random.uniform(-2, 2, 3)
    G_value = call_y_F1(x[0], x[1], x[2])
    while(G_value > 1.0e-10):
        grad = call_y_F1_prime_mat(x[0], x[1], x[2])
        x = x - (lr*grad).reshape((3,))
        G_value = call_y_F1(x[0], x[1], x[2])    
    return x


y_F2 = F2_curve(X)
y_F2_prime1 = y_F2.diff(x1)
y_F2_prime2 = y_F2.diff(x2)
y_F2_prime3 = y_F2.diff(x3)
y_F2_prime_mat = Matrix([y_F2_prime1,y_F2_prime2,y_F2_prime3])
call_y_F2_prime_mat = lambdify((x1, x2,x3), y_F2_prime_mat, 'numpy')
call_y_F2 = lambdify((x1, x2,x3), y_F2, 'numpy')


def generate_F2_sample():
    lr = 0.01
    # x = np.random.random(3)
    x = np.random.uniform(-2, 2, 3)
    G_value = call_y_F2(x[0], x[1], x[2])
    while(G_value > 1.0e-10):
        grad = call_y_F2_prime_mat(x[0], x[1], x[2])
        x = x - (lr*grad).reshape((3,))
        G_value = call_y_F2(x[0], x[1], x[2])    
    return x



def generate_sample():
    lr = 0.01
    # x = np.random.random(3)
    x = np.random.uniform(-2, 2, 3)
    G_value = call_y(x[0], x[1], x[2])
    while(G_value > 1.0e-10):
        grad = call_yprime_mat(x[0], x[1], x[2])
        x = x - (lr*grad).reshape((3,))
        G_value = call_y(x[0], x[1], x[2])    
    return x


