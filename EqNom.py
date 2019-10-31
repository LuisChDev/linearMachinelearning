import numpy as np  # type: ignore
from typing import List
from sklearn.model_selection import train_test_split  # type: ignore
from math import sqrt

valores = np.loadtxt('fish_length.txt', dtype=float)

# extrayendo el 70% de los datos
train, test = train_test_split(valores, test_size=0.3)

# extrayendo las columnas con X
Xs = train[:, [0, 1]]
# print(Xs)

# extrayendo las columnas con Y
Y = train[:, [2]]
# print(Y)

# # añadiendo la columna constante
ones = np.ones([len(train), 1])
Xs = np.concatenate((ones, Xs), axis=1)
# print(Xs)

# # Calculando la matriz de proyección
# # Es de la forma (X^t X)^-1 X^t
proy = np.dot(np.linalg.inv(np.dot(np.transpose(Xs), Xs)), np.transpose(Xs))
# print(proy)

# multiplicando las matrices para obtener la proyección sobre el espacio 3d
result = np.dot(proy, Y)
print("En su orden: b, m1 y m2.")
print(result)
input("presione Enter para continuar...")

Xs_test = test[:, [0, 1]]
Y_test = test[:, [2]]

# generando los valores predichos
Y_pred = np.array([result[0] + result[1]*x1 + result[2]*x2 for [x1, x2] in Xs_test])
print("Estos son los valores pronosticados para cada par de valores de X.")
print(Y_pred)
input("presione Enter para continuar...")

# calculando los errores
errores = np.subtract(Y_pred, Y_test)
err_2 = np.sum(np.square(errores))
# total = sqrt(np.sum(err_2))
print("Este es el error cuadrático incurrido por el modelo.")
print(err_2)
