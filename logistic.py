import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List

# variables:
epochs = 1000  # numero de iteraciones.
b = 0          # constante en la fórmula.
m1 = 0         # coeficiente del primer examen.
m2 = 0         # coeficiente del segundo examen.
alpha = 0.001  # tasa de aprendizaje.

# cargando el contenido del archivo:
valores = np.loadtxt("grades_data.txt", dtype=float, delimiter=',')
N = len(valores)

# separando la matriz de entrada en los vectores X y Y
B = np.ones((N, 1), dtype=float)
X1_ = valores[:, [0]]
X2_ = valores[:, [1]]
Y = valores[:, [2]]

# se normalizan los valores de los vectores.
X1 = (X1_ - np.mean(X1_))/(max(X1_) - min(X1_))
X2 = (X2_ - np.mean(X2_))/(max(X2_) - min(X2_))

# calculando la predicción de acuerdo a los valores iniciales.
Y_pred = (B * b) + (X1 * m1) + (X2 * m2)

# por cada iteración planeada, calcular el valor del gradiente y restarlo al
# valor actual de los parámetros.
for _ in range(epochs):
    # se calculan los gradientes
    b_grad = 0
    pass
