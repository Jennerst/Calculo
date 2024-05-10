#Diferencias Finitas hacia adelante
def funcion(x):
    resultado = pow(x, 3) + pow(x, 2) + x + 3
    return resultado

def derivada_adelante(x, h):
    deri = (funcion(x + h) - funcion(x)) / h
    return deri

def derivadao4_adelante(x, h):
    deri = (-funcion(x + 2 * h) + 4 * funcion(x + h) - 3 * funcion(x)) / (2 * h)
    return deri

x = float(input("En qué valor desea calcular la derivada? "))
h = float(input("Ingrese el valor de h: "))

print("Derivada hacia adelante de primer orden:", derivada_adelante(x, h))
print("Derivada hacia adelante de cuarto orden:", derivadao4_adelante(x, h))



#Diferencias Finitas hacia atras
def funcion(x):
    resultado = pow(x, 3) + pow(x, 2) + x + 3
    return resultado

def derivada_atras(x, h):
    deri = (funcion(x) - funcion(x - h)) / h
    return deri

def derivadao4_atras(x, h):
    deri = (funcion(x - 2 * h) - 4 * funcion(x - h) + 3 * funcion(x)) / (2 * h)
    return deri

x = float(input("En qué valor desea calcular la derivada? "))
h = float(input("Ingrese el valor de h: "))

print("Derivada hacia atrás de primer orden:", derivada_atras(x, h))
print("Derivada hacia atrás de cuarto orden:", derivadao4_atras(x, h))



#Diferencias Finitas Centrales
def funcion(x):
    resultado = pow(x,3)+pow(x,2)+x+3
    return resultado

def derivada (x):
    deri=(funcion(x+h)- funcion(x-h))/(2*h)
    return deri

def derivadao4(x):
    deri=(-funcion(x+h*2)+8*funcion(x+h)-8*funcion(x+h)+funcion(x-h*2))
    return deri

x=float(input("en que valor desea calcular la derivada? "))
h=float(input("ingrese el valor de h: "))

print(derivada(x))
print(derivadao4(x))

print("comprobacion")
from scipy.misc import derivate 
print(derivate(funcion,x,dx=le-6))
print('{0:.16f}'.format(derivate(funcion,x,dx=le-6)))



#Diferenciacion numerica por diferencias divididas
import numpy as np
def funcion(x):
    resultado = pow(x, 3) + pow(x, 2) + x + 3
    return resultado

def diferencias_divididas(x_data, y_data):
    n = len(x_data)
    F = np.zeros((n, n))

    for i in range(n):
        F[i, 0] = y_data[i]

    for j in range(1, n):
        for i in range(n - j):
            F[i, j] = (F[i + 1, j - 1] - F[i, j - 1]) / (x_data[i + j] - x_data[i])

    return F[0]
def derivada_diferencias_divididas(x_data, y_data, x):
    n = len(x_data)
    a = diferencias_divididas(x_data, y_data)
    result = a[n - 1]

    for i in range(n - 2, -1, -1):
        result = result * (x - x_data[i]) + a[i]

    return result
x_data = np.array([1, 2, 3, 4, 5])
y_data = funcion(x_data)
x_punto = 3.5

derivada = derivada_diferencias_divididas(x_data, y_data, x_punto)
print("La derivada en x =", x_punto, "es:", derivada)



#Metodo de Richardson
import numpy as np

def funcion(x):
    return x**3 + x**2 + x + 3

def derivada_central(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def richardson(f, x, h, n):
    D1 = derivada_central(f, x, h)
    D2 = derivada_central(f, x, h/2)
    
    return D1 + (D1 - D2) / (2**n - 1)

x = 3.5

h = 0.1

n = 2

derivada_richardson = richardson(funcion, x, h, n)

print("La derivada en x =", x, "calculada mediante el método de Richardson es:", derivada_richardson)



#Diferenciacion numerica por interpolacion polinomica
import numpy as np
from scipy.interpolate import lagrange

def funcion(x):
    return x**3 + x**2 + x + 3

x_data = np.array([1, 2, 3, 4, 5])
y_data = funcion(x_data)

poly_interp = lagrange(x_data, y_data)

def derivada(x):
    return poly_interp.deriv()(x)

x_punto = 3.5

derivada_en_x_punto = derivada(x_punto)

print("La derivada en x =", x_punto, "es:", derivada_en_x_punto)
