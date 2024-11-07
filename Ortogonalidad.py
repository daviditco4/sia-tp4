import numpy as np
from itertools import combinations

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
        letras[letra_actual] = np.array(matriz).flatten()  # Aplanar la matriz a un vector
        letra_actual = chr(ord(letra_actual) + 1)
        
        # Terminar si llegamos a la letra Z
        if letra_actual > 'Z':
            break
            
    return letras

def calcular_metricas_triangulo_superior(letras):
    """
    Calcula los promedios, máximos y frecuencias de los valores en el triángulo superior de las matrices de productos escalares
    entre cada par de letras de todos los grupos posibles de 4 letras.
    
    Args:
    - letras (dict): diccionario con letras como claves y matrices 5x5 como valores.
    
    Returns:
    - metricas (list): lista con las métricas calculadas para cada grupo de letras.
    """
    letras_nombres = list(letras.keys())
    
    # Generar todos los grupos de 4 letras
    grupos = list(combinations(letras_nombres, 4))
    
    metricas = []
    
    # Para cada grupo de 4 letras, calcular las métricas
    for grupo in grupos:
        # Crear una matriz de 4x4 para almacenar los productos escalares
        matriz_productos = np.zeros((4, 4), dtype=int)
        
        # Calcular los productos escalares y llenar la matriz
        for i in range(4):
            for j in range(4):
                if i == j:
                    matriz_productos[i, j] = 0  # La diagonal será siempre 0
                else:
                    matriz_productos[i, j] = np.dot(letras[grupo[i]], letras[grupo[j]])
        
        # Extraer el triángulo superior (sin la diagonal)
        triangulo_superior = matriz_productos[np.triu_indices(4, k=1)]
        
        # Calcular el promedio de los valores absolutos en el triángulo superior
        promedio = np.mean(np.abs(triangulo_superior))
        
        # Calcular el valor máximo absoluto y cuántas veces aparece en el triángulo superior
        max_val = np.max(np.abs(triangulo_superior))
        max_count = np.count_nonzero(np.abs(triangulo_superior) == max_val)
        
        metricas.append({
            'grupo': grupo,
            'promedio': promedio,
            'max_val': max_val,
            'max_count': max_count
        })
    
    return metricas

def ordenar_metricas(metricas):
    """
    Ordena las métricas de los grupos de letras por:
    1. Promedio ascendente
    2. Valor máximo ascendente
    3. Cantidad de veces que aparece el valor máximo ascendente
    
    Args:
    - metricas (list): lista con las métricas calculadas para cada grupo de letras.
    
    Returns:
    - metricas_ordenadas (list): lista de las métricas ordenadas.
    """
    return sorted(metricas, key=lambda x: (x['promedio'], x['max_val'], x['max_count']))

def mostrar_top_bottom(metricas_ordenadas):
    """
    Muestra el top 20 y bottom 20 de los grupos de letras ordenados.
    
    Args:
    - metricas_ordenadas (list): lista de las métricas ordenadas.
    """
    print("\nTop 20 grupos de letras:")
    for i, m in enumerate(metricas_ordenadas[:20]):
        print(f"{i+1}. {m['grupo']} | Promedio: {m['promedio']:.4f} | Max: {m['max_val']} | Frecuencia: {m['max_count']}")
    
    print("\nBottom 20 grupos de letras:")
    for i, m in enumerate(metricas_ordenadas[-20:]):
        print(f"{len(metricas_ordenadas) - 19 + i}. {m['grupo']} | Promedio: {m['promedio']:.4f} | Max: {m['max_val']} | Frecuencia: {m['max_count']}")

# Cargar las letras desde el archivo
filename = 'letras.txt'  # Nombre del archivo que contiene las matrices
letras = cargar_letras(filename)

# Calcular las métricas y ordenar los grupos
metricas = calcular_metricas_triangulo_superior(letras)
metricas_ordenadas = ordenar_metricas(metricas)

# Mostrar el top 20 y bottom 20
mostrar_top_bottom(metricas_ordenadas)
