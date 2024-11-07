import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def cargar_letras(filename):
    """
    Carga letras en formato de matrices 5x5 desde un archivo de texto.
    
    Args:
    - filename (str): nombre del archivo de texto que contiene las matrices.
    
    Returns:
    - letras (dict): un diccionario con cada letra y su matriz 5x5.
    """
    letras = {}
    letra_actual = 'A'
    
    with open(filename, 'r') as file:
        lineas = file.readlines()
    
    # Procesar cada letra en secuencia
    for i in range(0, len(lineas), 5):
        # Obtener cinco líneas para una letra y convertirlas en matriz
        matriz = [list(map(int, line.strip().split(','))) for line in lineas[i:i+5]]
        letras[letra_actual] = np.array(matriz)
        
        # Avanzar a la siguiente letra
        letra_actual = chr(ord(letra_actual) + 1)
        
        # Terminar si llegamos a la letra Z
        if letra_actual > 'Z':
            break
            
    return letras

def graficar_letras(letras):
    """
    Grafica las matrices de letras en formato 5x5.
    
    Args:
    - letras (dict): diccionario con letras como claves y matrices 5x5 como valores.
    """
    # Crear un colormap personalizado: -1 en blanco y 1 en azul
    cmap = ListedColormap(['white', 'blue'])
    
    # Definir una cuadrícula de 5x6 para las 26 letras
    fig, axs = plt.subplots(5, 6, figsize=(12, 10))
    fig.suptitle('Alfabeto en formato de matrices 5x5', fontsize=16)

    # Configuración de la cuadrícula
    for idx, (letra, matriz) in enumerate(letras.items()):
        fila = idx // 6
        col = idx % 6
        ax = axs[fila, col]
        
        # Mostrar cada letra como una imagen en blanco y azul
        ax.imshow(matriz, cmap=cmap, vmin=-1, vmax=1)
        ax.set_title(letra)
        ax.axis('off')  # Ocultar ejes para claridad

    # Ocultar cualquier subplot restante (en caso de menos de 30 letras)
    for idx in range(len(letras), 30):
        fig.delaxes(axs[idx // 6, idx % 6])

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# Cargar las letras desde el archivo y graficarlas
filename = 'letras.txt'  # Nombre del archivo que contiene las matrices
letras = cargar_letras(filename)
graficar_letras(letras)
