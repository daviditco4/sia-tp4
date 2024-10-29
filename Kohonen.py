import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_neuron_hit_map(som, data):
    """
    Dibuja un mapa que muestra la cantidad de ejemplos asignados a cada neurona.
    :param som: Red SOM entrenada
    :param data: Datos normalizados
    """
    # Crear una matriz de ceros con el tamaño de la red SOM (m x n)
    hit_map = np.zeros((som.m, som.n))
    
    # Mapeamos cada ejemplo de datos a su neurona ganadora (BMU)
    for sample in data:
        bmu_idx = som._find_bmu(sample)
        hit_map[bmu_idx] += 1  # Aumentamos el contador en la neurona ganadora
    
    # Graficamos el mapa de hits
    plt.figure(figsize=(8, 8))
    plt.imshow(hit_map, cmap='Blues', origin='upper')
    plt.colorbar(label='Número de ejemplos')
    plt.title('Cantidad de registros asignados a cada neurona')
    
    # Añadir el número de ejemplos sobre cada neurona
    for i in range(som.m):
        for j in range(som.n):
            plt.text(j, i, int(hit_map[i, j]), ha='center', va='center', color='black')

    plt.show()

def plot_countries_on_som(som, data, country_names):
    """
    Dibuja los nombres de los países en el mapa SOM.
    :param som: Red SOM entrenada
    :param data: Datos normalizados
    :param country_names: Lista con los nombres de los países
    """
    plt.figure(figsize=(10, 10))
    
    # Obtenemos las posiciones de cada país en la red SOM
    mappings = som.map_data(data)
    
    # Graficar los nombres de los países en las neuronas ganadoras
    for position, countries_indices in mappings.items():
        # Obtener los nombres de los países correspondientes a las posiciones
        country_list = [country_names[i] for i in countries_indices]
        # Posición en el gráfico
        plt.text(position[0], position[1], '\n'.join(country_list), fontsize=8, ha='center', va='center')
    
    # Configurar la cuadrícula
    plt.xlim(0, som.m)
    plt.ylim(0, som.n)
    plt.title('Países en la Red SOM')
    plt.grid(True)
    plt.show()


class KohonenSOM:
    def __init__(self, m, n, dim, learning_rate=0.5, radius=None, radius_decay=0.99, lr_decay=0.99):
        """
        Inicializa la red de Kohonen (SOM)
        :param m: Número de filas de la red
        :param n: Número de columnas de la red
        :param dim: Dimensiones de los datos de entrada
        :param learning_rate: Tasa de aprendizaje inicial
        :param radius: Radio inicial de vecindad
        :param radius_decay: Factor de decaimiento del radio
        :param lr_decay: Factor de decaimiento de la tasa de aprendizaje
        """
        self.m = m
        self.n = n
        self.dim = dim
        self.learning_rate = learning_rate
        self.radius = radius if radius else max(m, n) / 2  # Radio inicial
        self.radius_decay = radius_decay
        self.lr_decay = lr_decay
        
        # Inicializamos los pesos de las neuronas de manera aleatoria
        self.weights = np.random.rand(m, n, dim)

    def _find_bmu(self, sample):
        """
        Encuentra la neurona ganadora (BMU) para una muestra dada.
        :param sample: Muestra de entrada
        :return: Coordenadas de la BMU
        """
        # Calcular las distancias euclidianas entre el sample y todas las neuronas
        bmu_idx = np.argmin(np.linalg.norm(self.weights - sample, axis=-1))
        return np.unravel_index(bmu_idx, (self.m, self.n))

    def _update_weights(self, sample, bmu_idx, iteration, total_iterations):
        """
        Actualiza los pesos de la red de Kohonen
        :param sample: Muestra de entrada
        :param bmu_idx: Índice de la neurona ganadora
        :param iteration: Iteración actual
        :param total_iterations: Número total de iteraciones
        """
        # Decaimiento de la tasa de aprendizaje y el radio de vecindad
        lr = self.learning_rate * (1 - iteration / total_iterations)
        radius = max(1, self.radius * (1 - iteration / total_iterations))
        
        for x in range(self.m):
            for y in range(self.n):
                # Calcular la distancia euclidiana entre la neurona y la BMU
                dist = np.linalg.norm(np.array([x, y]) - np.array(bmu_idx))
                
                # Si la neurona está dentro del radio de influencia
                if dist <= radius:
                    # Actualizar los pesos de la neurona
                    self.weights[x, y] += lr * (sample - self.weights[x, y])

    def train(self, data, num_iterations):
        """
        Entrena la red SOM
        :param data: Conjunto de datos de entrada
        :param num_iterations: Número de iteraciones de entrenamiento
        """
        for iteration in range(num_iterations):
            # Seleccionar una muestra aleatoria
            sample = data[np.random.randint(0, len(data))]
            
            # Encontrar la neurona ganadora (BMU)
            bmu_idx = self._find_bmu(sample)
            
            # Actualizar los pesos de la red
            self._update_weights(sample, bmu_idx, iteration, num_iterations)

    def map_data(self, data):
        """
        Mapea las muestras de entrada a las posiciones de la red.
        :param data: Conjunto de datos de entrada
        :return: Mapeo de las muestras en la red
        """
        mappings = {}
        for i, sample in enumerate(data):
            bmu_idx = self._find_bmu(sample)
            if bmu_idx not in mappings:
                mappings[bmu_idx] = []
            mappings[bmu_idx].append(i)
        return mappings

    def distance_map(self):
        """
        Calcula las distancias promedio entre las neuronas vecinas
        :return: Mapa de distancias promedio
        """
        dist_map = np.zeros((self.m, self.n))
        for x in range(self.m):
            for y in range(self.n):
                neighbors = []
                if x > 0: neighbors.append(self.weights[x - 1, y])
                if x < self.m - 1: neighbors.append(self.weights[x + 1, y])
                if y > 0: neighbors.append(self.weights[x, y - 1])
                if y < self.n - 1: neighbors.append(self.weights[x, y + 1])
                
                dist_map[x, y] = np.mean([np.linalg.norm(self.weights[x, y] - neighbor) for neighbor in neighbors])
        return dist_map

# Cargar el archivo CSV
data = pd.read_csv('europe.csv')

# Seleccionar solo las columnas numéricas para entrenar la red
data_numeric = data[['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']]

# Normalizar los datos
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data_numeric)

dimension=[3, 4]

for i in range(len(dimension)):
    # Entrenamiento de la red SOM
    som = KohonenSOM(m=dimension[i], n=dimension[i], dim=7, learning_rate=0.3)

    # Supongamos que ya cargamos y normalizamos los datos en 'data_normalized'
    som.train(data_normalized, num_iterations=10000)

    # Graficar el mapa de distancias
    plt.figure(figsize=(7, 7))
    plt.pcolor(som.distance_map())
    plt.colorbar()
    plt.title('Distancias promedio entre neuronas vecinas')
    plt.show()

    # Mapeo de los países en la red
    mappings = som.map_data(data_normalized)

    # Ejecutar el gráfico
    plot_countries_on_som(som, data_normalized, data['Country'])

    # Llamamos a la función para generar el gráfico
    plot_neuron_hit_map(som, data_normalized)