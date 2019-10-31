# Regresión lineal

Este módulo permite calcular los coeficientes de mejor ajuste lineal para un
conjunto de datos.  

Los datos deben estar disponibles en un archivo. de las `n`
columnas en el archivo (separadas por espacios), las primeras `n-1` se
consideran variables de entrada y la última es el valor de salida.  

Una vez cargados, los datos se cargan a pandas, y se emplea la biblioteca
scikit-learn para generar el modelo.
