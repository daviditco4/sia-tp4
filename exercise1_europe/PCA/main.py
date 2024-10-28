import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def standard_dev_matrix(mat, prom):
	toRet = []
	for col in range(len(mat[0])):
		sumando = 0
		promedio_col = prom[col]
		for fila in range((len(mat))):
			sumando += ((mat[fila][col] - promedio_col) ** 2)
		varianza = sumando / (len(mat) - 1)
		deviation = np.sqrt(varianza)
		toRet.append(deviation)
	return toRet

def average_matrix(mat):
	toRet = []
	for col in range(len(mat[0])):
		sum = 0
		for fila in range(len(mat)):
			sum += mat[fila][col]
		toRet.append(float(sum / len(mat)))
	return toRet

def estandarize_matrix(mat, prom, std):
	toRet = np.zeros((mat.shape[0], mat.shape[1]))
	for row in range(len(mat)):
		for col in range(len(mat[0])):
			est_val = mat[row][col] - prom[col]
			est_val = float(est_val / std[col])
			toRet[row][col] = est_val
	return toRet
 
np.set_printoptions(suppress=True)

#Cargar el archivo europe.csv
df = pd.read_csv('europe.csv')

#Mostrar la data
#print(df)
#Le saco la columna con los nombres de los paises
df_sin_nombre_pais = df.drop(df.columns[0], axis=1)

#Cantidad de Variables y Muestras de datos
N = df_sin_nombre_pais.shape[1]
M = df_sin_nombre_pais.shape[0]

#Creo la matriz de data, de estandarizacion y de correlacion
data_matrix = np.zeros((M,N))
correlation_matrix = np.zeros((N,N))

#Itero por cada columna y guardo los valores en la matriz
#valores = df_sin_nombre_pais.to_numpy().tolist()
n_col = 0
for index, row in df_sin_nombre_pais.iterrows():
	for value in row:
		data_matrix[index][n_col] = (float)(value)
		n_col += 1
	n_col = 0
	
#Calculo el promedio y el desvio estandar de cada variable (columna)
promedio_var = average_matrix(data_matrix)
std_var = standard_dev_matrix(data_matrix, promedio_var)
std_var = [float(val) for val in std_var]
promedio_var2 = np.mean(data_matrix, axis=0)
std_var2 = np.std(data_matrix, axis=0, ddof=1)

#Calculo la matrix estandarizada
est_matrix = estandarize_matrix(data_matrix, promedio_var, std_var)

#Realizo los boxplot de las variables antes de la estandarizacion
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_matrix)

# Cambiar los nombres de las etiquetas del eje X
column_names = ["Area", "GDP", "Inflation", "Life.expect", "Military", "Pop.growth", "Unemployment"]
plt.xticks(ticks=range(len(column_names)), labels=column_names)

# Añadir título y etiquetas
plt.title('Boxplot de las Variables Originales previo a Estandarizacion')

plt.xlabel('Variables')
plt.ylabel('Valores')

# Mostrar el gráfico
plt.show()

#Realizo los boxplot de las variables despues de la estandarizacion
plt.figure(figsize=(10, 6))
sns.boxplot(data=est_matrix)

# Cambiar los nombres de las etiquetas del eje X
column_names = ["Area", "GDP", "Inflation", "Life.expect", "Military", "Pop.growth", "Unemployment"]
plt.xticks(ticks=range(len(column_names)), labels=column_names)

# Añadir título y etiquetas
plt.title('Boxplot de las Variables Originales despues de la Estandarizacion')
plt.xlabel('Variables')
plt.ylabel('Valores')

# Mostrar el gráfico
plt.show()

#Realizando el biplot de PCA1 y PCA2

# Realizar PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(est_matrix)

# Graficar PCA1 y PCA2
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], color='blue', label='Puntos')

# Añadir los nombres de los países a los puntos
for i, country in enumerate(df['Country']):
 
plt.text(X_pca[i, 0], X_pca[i, 1], country, color='blue', fontsize=12, ha='right')

# Añadir los vectores de las variables originales
# Las cargas de las variables están en pca.components_
for i, col in enumerate(df_sin_nombre_pais.columns):
 
plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], 
 
color='red', alpha=0.5, head_width=0.05, head_length=0.1)
 
plt.text(pca.components_[0, i] * 1.1, pca.components_[1, i] * 1.1, col, color='red', fontsize=12)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA con vectores de variables originales')
plt.grid(True)
plt.axis('equal')
plt.show()

#Haciendo Biplot del indice a partir del PCA1
# Realizar PCA
pca = PCA(n_components=1) # Solo PCA1
X_pca = pca.fit_transform(est_matrix)

# Graficar el biplot (solo PCA1)
plt.figure(figsize=(10, 6))

# Puntos de los países proyectados en PCA1
plt.scatter(df['Country'], X_pca[:, 0], color='blue', s=100)

# Añadir nombres de los países al gráfico
for i, country in enumerate(df['Country']):
 
plt.text(country, X_pca[i, 0], country, color='blue', fontsize=10, ha='right')

# Configurar el gráfico
plt.xlabel('Países')
plt.ylabel('Valor de PCA 1')
plt.title('Biplot de PCA1 por Países')
plt.xticks(rotation=45) # Rotar etiquetas de países para mejor legibilidad
plt.grid(True)
plt.show()


#Hacer BoxPlot de los loadings de PCA1
# Crear el modelo PCA
pca = PCA(n_components=1) # Solo nos interesa el primer componente
pca.fit(est_matrix)

# Los loadings del PCA1 son los coeficientes (pesos) del componente principal
loadings_pca1 = pca.components_[0]

# Variables
variables = ["Area", "GDP", "Inflation", "Life.expect", "Military", "Pop.growth", "Unemployment"]

# Colores para las barras
colors = ['green' if loading > 0 else 'red' for loading in loadings_pca1]

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))
bars = plt.bar(variables, loadings_pca1, color=colors)

# Añadir etiquetas
plt.xlabel('Variables')
plt.ylabel('Loading en PCA1')
plt.title('Loadings de PCA1 para diferentes variables')

# Mostrar el gráfico
plt.tight_layout()
plt.show()


#Hacer BarPlot de los loadings de PCA2
# Crear el modelo PCA
pca = PCA(n_components=2) # Solo nos interesa el primer componente
pca.fit(est_matrix)

# Los loadings del PCA1 son los coeficientes (pesos) del componente principal
loadings_pca2 = pca.components_[1]

# Variables
variables = ["Area", "GDP", "Inflation", "Life.expect", "Military", "Pop.growth", "Unemployment"]

# Colores para las barras
colors = ['green' if loading > 0 else 'red' for loading in loadings_pca2]

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))
bars = plt.bar(variables, loadings_pca2, color=colors)

# Añadir etiquetas
plt.xlabel('Variables')
plt.ylabel('Loading en PCA2')
plt.title('Loadings de PCA2 para diferentes variables')

# Mostrar el gráfico
plt.tight_layout()
plt.show()