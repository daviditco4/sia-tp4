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
        print(self.weights)

    def predict(self, noisy_pattern, steps=5):
        current_state = noisy_pattern.flatten()
        history = [current_state.copy()]

        for step in range(steps):
            for i in range(self.size):
                activation = np.dot(self.weights[i], current_state)
                current_state[i] = 1 if activation > 0 else -1
            
            history.append(current_state.copy())
            
            # Comprobar si el estado es estable (comparar con el estado anterior)
            if np.array_equal(history[-1], history[-2]):
                #print(f"Estado estable alcanzado en el paso {step + 1}.")
                break  # Salir del bucle si el estado es estable
        
        return history
    
    def is_spurious_state(self, final_pattern):
        # Comprobar si el patrón final coincide con alguno de los patrones originales
        for original_pattern in self.patterns:
            if np.array_equal(final_pattern, original_pattern):
                return False  # No es espurio (coincide con un patrón almacenado)
        return True  # Es espurio (no coincide con ningún patrón almacenado)
    
    def identify_pattern(self, final_pattern, patterns):
        # Identificar si el patrón final coincide con alguno de los almacenados
        for i, original_pattern in enumerate(patterns):
            if np.array_equal(final_pattern, original_pattern):
                return i  # Devuelve el índice del patrón coincidente
        return None  # Si no coincide con ninguno

    @staticmethod
    def add_noise(pattern, noise_level):
        noisy_pattern = pattern.copy()
        num_noisy_elements = int(noise_level * pattern.size)
        indices = np.random.choice(pattern.size, num_noisy_elements, replace=False)
        noisy_pattern.ravel()[indices] *= -1  # Invert some values
        return noisy_pattern

    @staticmethod
    def plot_patterns(patterns, title):
        if len(patterns) == 1:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(patterns[0].reshape(5, 5), cmap='gray_r', vmin=-1, vmax=1)  # Invert colors
            ax.axis('off')
        else:
            fig, axes = plt.subplots(1, len(patterns), figsize=(15, 5))
            for ax, pattern in zip(axes, patterns):
                ax.imshow(pattern.reshape(5, 5), cmap='gray_r', vmin=-1, vmax=1)  # Invert colors
                ax.axis('off')
        
        plt.suptitle(title)
        plt.show()

# Función para introducir ruido en un patrón
def add_noise2(pattern, noise_level=0.1):
    noisy_pattern = pattern.copy().flatten()
    num_flips = int(noise_level * noisy_pattern.size)
    flip_indices = random.sample(range(noisy_pattern.size), num_flips)
    
    for idx in flip_indices:
        noisy_pattern[idx] = -noisy_pattern[idx]  # Invertir el bit (de -1 a 1 o viceversa)
    
    return noisy_pattern.reshape(pattern.shape)
        
# Función para correr los experimentos y calcular las probabilidades
def run_experiments(hopfield_net, original_pattern, patterns, noise_level=0.1, num_experiments=100):
    converged_counts = np.zeros(len(patterns) + 1)  # Para contar convergencias a cada patrón y espurios
    
    for _ in range(num_experiments):
        noisy_pattern = add_noise2(original_pattern, noise_level=noise_level)
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

# Define the original patterns for A, J, M, L using -1 and 1
patterns = [
    np.array([[1,  1,  1,  1, 1],
              [1, -1, -1, -1, 1],
              [1,  1,  1,  1, 1],
              [1, -1, -1, -1, 1],
              [1, -1, -1, -1, 1]]),  # A
    np.array([[ 1, 1,  1,  1,  1],
              [-1, -1, 1, -1, -1],
              [-1, -1, 1, -1, -1],
              [ 1, -1, 1, -1, -1],
              [ 1,  1, 1, -1, -1]]),  # J
    np.array([[1, -1, -1, -1, 1],
              [1,  1, -1,  1, 1],
              [1, -1,  1, -1, 1],
              [1, -1, -1, -1, 1],
              [1, -1, -1, -1, 1]]),  # M
    np.array([[1, -1, -1, -1, -1],
              [1, -1, -1, -1, -1],
              [1, -1, -1, -1, -1],
              [1, -1, -1, -1, -1],
              [1, 1, 1, 1, 1]])   # L
]

letras = ['A', 'J', 'M', 'L']
noise_levels = [0.1, 0.3, 0.5]

# Initialize the Hopfield Network
hopfield_net = HopfieldNetwork(size=25)
hopfield_net.train(patterns)

for i in range(len(noise_levels)):
    # Create a noisy version of one of the patterns (e.g., A)
    noisy_pattern0 = HopfieldNetwork.add_noise(patterns[0], noise_level=noise_levels[i])
    noisy_pattern1 = HopfieldNetwork.add_noise(patterns[1], noise_level=noise_levels[i])
    noisy_pattern2 = HopfieldNetwork.add_noise(patterns[2], noise_level=noise_levels[i])
    noisy_pattern3 = HopfieldNetwork.add_noise(patterns[3], noise_level=noise_levels[i])

    # Show the noisy pattern
    #HopfieldNetwork.plot_patterns([noisy_pattern], title="Noisy Pattern")

    # Recover the pattern using the Hopfield Network
    history0 = hopfield_net.predict(noisy_pattern0, steps=10)
    history1 = hopfield_net.predict(noisy_pattern1, steps=10)
    history2 = hopfield_net.predict(noisy_pattern2, steps=10)
    history3 = hopfield_net.predict(noisy_pattern3, steps=10)

    # Show each step in the recovery process
    HopfieldNetwork.plot_patterns(history0, title="Recovery Steps")
    HopfieldNetwork.plot_patterns(history1, title="Recovery Steps")
    HopfieldNetwork.plot_patterns(history2, title="Recovery Steps")
    HopfieldNetwork.plot_patterns(history3, title="Recovery Steps")

for j in range(len(noise_levels)):
    for k in range(len(patterns)):
        print(f"Porcentajes de convergencia para la letra inicial {letras[k]}, with noise level {noise_levels[j]}")
        # Patrón original para realizar las pruebas (por ejemplo, la letra A)
        original_pattern = patterns[k]

        # Ejecutar los experimentos
        percentages = run_experiments(hopfield_net, original_pattern, patterns, noise_level=noise_levels[j], num_experiments=1000)

        # Mostrar los resultados
        for i, pattern in enumerate(patterns):
            print(f"Porcentaje de convergencia a la letra {letras[i]}: {percentages[i]:.2f}%")
        print(f"Porcentaje de convergencia a un estado espurio: {percentages[-1]:.2f}%")