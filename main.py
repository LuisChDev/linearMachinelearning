import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List

# variables:
num_iters = 10000
b = 1337.0
m1 = 26.0
m2 = -17.0
alpha = 0.00001

# cargando el contenido del archivo:
valores = np.loadtxt("fish_length.txt", dtype=float)
N = float(len(valores))
# print(valores)

# separando la matriz de entrada en los vectores X y Y
B = np.ones((len(valores), 1), dtype=float)
X1 = valores[:, [0]]
X2 = valores[:, [1]]
Y = valores[:, [2]]

# se calcula el error de estos valores
Y_pred = B + (X1 * m1) + (X2 * m2)
error = sum(np.square(Y_pred - Y))

print("El error con la formulación inicial es: ", error)

# por cada iteración:
for _ in range(num_iters):
    # se calculan las predicciones actuales
    Y_pred = (B * b) + (X1 * m1) + (X2 * m2)
    # se calculan los gradientes
    b_grad = -(2/N) * sum(Y - Y_pred)
    m1_grad = -(2/N) * sum(X1 * (Y - Y_pred))
    m2_grad = -(2/N) * sum(X2 * (Y - Y_pred))
    # se recalculan los valores de los parámetros
    b = b - (alpha * b_grad)
    m1 = m1 - (alpha * m1_grad)
    m2 = m2 - (alpha * m2_grad)


print("Los valores calculados son: ")
print("b: ", b)
print("m1: ", m1)
print("m2: ", m2)

# se calcula el error de este resultado
Y_pred = B + (X1 * m1) + (X2 * m2)
error = sum(np.square(Y_pred - Y))

print("El error calculado es: ", error)
