import numpy as np
import random
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for pattern in patterns:
            pattern = pattern.flatten()
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)
        self.weights /= self.size

    def energy(self, state):
        """
        Calcula la función de energía de la red para un estado dado.
        
        Args:
        - state (ndarray): El vector de estado actual de la red.
        
        Returns:
        - energy (float): El valor de la función de energía.
        """
        return -0.5 * np.sum(np.outer(state, state) * self.weights)

    def predict(self, noisy_pattern, steps=5):
        current_state = noisy_pattern.flatten()
        energies = [self.energy(current_state)]  # Guardar la energía inicial

        for step in range(steps):
            for i in range(self.size):
                activation = np.dot(self.weights[i], current_state)
                current_state[i] = 1 if activation > 0 else -1
            
            energies.append(self.energy(current_state))  # Guardar la energía después de cada paso
        
        return energies

    @staticmethod
    def add_noise(pattern, noise_level):
        noisy_pattern = pattern.copy()
        num_noisy_elements = int(noise_level * pattern.size)
        indices = np.random.choice(pattern.size, num_noisy_elements, replace=False)
        noisy_pattern.ravel()[indices] *= -1  # Invertir algunos valores
        return noisy_pattern

# Función para correr los experimentos y calcular las probabilidades
def run_experiments(hopfield_net, original_pattern, patterns, noise_level=0.1, num_experiments=100):
    converged_counts = np.zeros(len(patterns) + 1)  # Para contar convergencias a cada patrón y espurios
    
    for _ in range(num_experiments):
        noisy_pattern = HopfieldNetwork.add_noise(original_pattern, noise_level=noise_level)
        final_pattern = hopfield_net.predict(noisy_pattern)[-1].tolist()
        final_pattern = np.reshape(final_pattern, (5,5))
        
        # Identificar el patrón al que ha convergido
        final_index = hopfield_net.identify_pattern(final_pattern, patterns)
        
        if final_index is None:
            # Estado espurio
            converged_counts[-1] += 1
        else:
            # Convergió a un patrón almacenado
            converged_counts[final_index] += 1
    
    # Calcular porcentajes
    converged_percentages = (converged_counts / num_experiments) * 100
    return converged_percentages

patterns = [
    np.array([[1,  1,  1,  1, 1],
              [1, -1, -1, -1,-1],
              [1, -1,  1,  1, 1],
              [1, -1, -1, -1, 1],
              [1,  1,  1,  1, 1]]),  # G
    np.array([[ 1, -1, -1, 1, -1],
              [ 1, -1, 1, -1, -1],
              [ 1, 1, -1, -1, -1],
              [ 1, -1, 1, -1, -1],
              [ 1, -1, -1, 1, -1]]),  # K
    np.array([[ 1,  1, 1,  1, 1],
              [-1, -1, 1, -1, -1],
              [-1, -1, 1, -1, -1],
              [-1, -1, 1, -1, -1],
              [-1, -1, 1, -1, -1]]),  # T
    np.array([[ 1, -1, -1, -1,  1],
              [ 1, -1, -1, -1,  1],
              [ 1, -1, -1, -1,  1],
              [-1,  1, -1,  1, -1],
              [-1, -1,  1, -1, -1]])   # V
]

pattern_bottom = [
    np.array([[1,  1,  1,  1, -1],
              [1, -1, -1, -1, 1],
              [1, -1,  -1,  -1, 1],
              [1, -1, -1, -1, 1],
              [1,  1,  1,  1, -1]]),  # D
    np.array([[1,  1,  1,  1, 1],
              [1, -1, -1, -1,-1],
              [1, -1,  1,  1, 1],
              [1, -1, -1, -1, 1],
              [1,  1,  1,  1, 1]]),  # G
    np.array([[ 1,  1,  1,  1, 1],
              [ 1, -1, -1, -1, 1],
              [ 1, -1, -1, -1, 1],
              [ 1, -1, -1, -1, 1],
              [ 1,  1,  1,  1, 1]]),  # O
    np.array([[ 1,  1,  1,  1, 1],
              [ 1, -1, -1, -1, 1],
              [ 1, -1,  1, -1, 1],
              [ 1, -1, -1,  1, 1],
              [ 1,  1,  1,  1, 1]])   # Q
]

letras = ['G', 'K', 'T', 'V']
letras_bottom = ['D', 'G', 'O', 'Q']
noise_levels = [0.1, 0.3, 0.5]
n_iterations = 1000  # Número de iteraciones

test_letters = letras_bottom #letras para mejor caso, letras_bottom para peor
test_patterns = pattern_bottom #patterns para mejor caso, pattern_bottom para peor

# Initialize the Hopfield Network
hopfield_net = HopfieldNetwork(size=25)
hopfield_net.train(test_patterns)

# Almacenar las energías promedio por letra y nivel de ruido
energy_stats = {letra: {noise: {'mean': []} for noise in noise_levels} for letra in test_letters}

# Colores predefinidos para cada línea
colores = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray']

# Realizar múltiples iteraciones
for noise_level in noise_levels:
    for k, letra in enumerate(test_letters):
        print(f"Procesando letra {letra} con ruido de nivel {noise_level}")

        # Almacenar las energías de todas las iteraciones
        all_energies = []

        for _ in range(n_iterations):
            noisy_pattern = HopfieldNetwork.add_noise(test_patterns[k], noise_level=noise_level)
            
            # Recuperar el patrón utilizando la red de Hopfield y obtener la energía
            energies = hopfield_net.predict(noisy_pattern, steps=10)
            
            # Guardamos las energías por cada paso en la lista all_energies
            all_energies.append(energies)
        
        # Aseguramos que todas las listas de energías tengan la misma longitud
        all_energies = np.array(all_energies)

        # Calcular el promedio de las energías en cada paso
        mean_energies = np.mean(all_energies, axis=0)

        # Guardar los resultados
        energy_stats[letra][noise_level]['mean'] = mean_energies

# Crear gráficos separados para cada nivel de ruido
for noise_level in noise_levels:
    plt.figure(figsize=(8, 6))
    for i, letra in enumerate(test_letters):
        # Elegir un color distinto para cada letra
        color = colores[i]
        # Graficar promedio de energía para la letra y nivel de ruido actual
        plt.plot(energy_stats[letra][noise_level]['mean'], label=f'{letra} - Promedio', linestyle='-', marker='o', color=color)

    # Configurar el eje x para que avance de 1 en 1
    plt.xticks(range(len(mean_energies)))

    plt.xlabel('Pasos')
    plt.ylabel('Energía')
    plt.title(f'Promedio de la Energía durante la Recuperación para el Nivel de Ruido {noise_level}')
    plt.legend()
    plt.grid(True)
    plt.show()
