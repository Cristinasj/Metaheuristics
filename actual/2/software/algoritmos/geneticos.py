import numpy as np
from algoritmos.evaluacionMatriz import *
from typing import Callable
import random 

def cruce_BLX(c1: np.array, c2: np.array) -> tuple:
    '''
    param: dos cromosomas 
    return: los dos cromosomas cruzados con BLX
    '''
    salida1 = np.zeros(len(c1)+1)
    salida2 = np.zeros(len(c1)+1)

    for i in range(0,len(c1)):

        if (c1[i] < c2[i]):
            cmin = c1[i]
            cmax = c2[i]
        else:
            cmin = c2[i]
            cmax = c1[i]
    l = cmax-cmin
    b1 = cmin-l*0.3
    if (b1 < 0):
      b1 = 0
    b2 = cmax+l*0.3
    if (b2 > 1):
      b2 = 1
    salida1[i] = np.random.uniform(b1, b2)
    salida2[i] = np.random.uniform(b1, b2)

    return salida1, salida2

def cruce_CA(c1: np.array, c2: np.array) -> tuple: 
    alfas = np.random.rand(len(c1)) 
    d1 = (1 - alfas) * c1 + alfas * c2
    d2 = alfas * c1 + (1- alfas) * c2
    c1[:] = d1
    c2[:] = d2
    return c1, c2

def mutacion(i, j, poblacion):
  poblacion[i][j] += np.random.normal(0, 0.3)
  if (poblacion[i][j] < 0):
    poblacion[i][j] = 0
  if (poblacion[i][j] > 1):
    poblacion[i][j] = 1

def torneoBinario(poblacion : np.array, seleccion: tuple) -> np.array: 
    '''
    param: 
        poblacion: matriz con la población y su valor de evaluación 
        seleccion: tupla con los indices de los elementos a competir
    return: 

    '''
    c1 = poblacion[seleccion[0]]
    c2 = poblacion[seleccion[1]]
    if (c1[-1] > c2[-1]):
        s = seleccion[0]
    else:
        s = seleccion[1]
    return s 

def torneoTetra(seleccion : tuple) -> tuple:
    '''
    param: 
        seleccion: 
            tupla de 4 elementos con su valor de evaluación
    return:
        tupla con las copias de los dos mejores elementos  
    ''' 
    if (seleccion[0][-1] > seleccion[1][-1]):
        c1 = seleccion[0]
    else:
        c1 = seleccion[1]

    if (seleccion[2][-1] > seleccion[3][-1]):
        c2 = seleccion[2]
    else:
        c2 = seleccion[3]

    return c1[0:-1], c2[0:-1] 

def generacional(elementos: list, tipo_cruce: Callable[[np.array, np.array], tuple], max_evaluaciones: int, p_mutacion: float, p_cruce: float) -> np.array:
    pass

def estacionario(matriz_datos: np.array, etiquetas: np.array, tipo_cruce: Callable[[np.array, np.array], tuple], max_evaluaciones: int, p_mutacion: float) -> np.array:
    '''
    param: 
        elementos: lista de elementos
        tipo_cruce: BLX o CA 
        max_iter: máximo número de iteraciones  
        p_mutacion: probabilidad de mutación  
    return: 
        vector de pesos optimizado con un algoritmo genético estacionario y el cruce indicado por parámetro 
    '''
    t_poblacion = len(matriz_datos)
    tam_w = len(matriz_datos[0])

    poblacion = np.zeros((t_poblacion, len(matriz_datos[0])+1))
    # Creación de la poblacion inicial
    # En la ultima posición está el valor de la función evaluación 
    for i in range(0, t_poblacion):
        w = np.random.rand(len(matriz_datos[0]))
        poblacion[i][0:len(matriz_datos[0])] = w
        poblacion[i][len(matriz_datos[0])] = funcionEvaluacionLeaveOneOut(matriz_datos, etiquetas, w)

    # Se ordena la población
    poblacion = np.array(poblacion)
    poblacion = poblacion[poblacion[:, -1].argsort()]

    iterations = 0
    while (iterations < max_evaluaciones):
        # Competición entre 4 individuos
        seleccion = poblacion[random.sample(range(t_poblacion), 4)]
        c1, c2 = torneoTetra(seleccion)

        # Se hacen los cruces
        seleccion = np.zeros((4, len(matriz_datos[0])+1))
        h1, h2 = tipo_cruce(c1, c2)

        # Se aplican las mutaciones
        mutaciones = [random.sample(range(tam_w*2), (int)(2*tam_w*p_mutacion))]
        for i in range(0, len(mutaciones)):
            r = np.random.randint(tam_w*2)
            i = r // tam_w
            j = r % tam_w
            mutacion(i, j, [h1, h2])

        h1[len(h1)-1] = funcionEvaluacionLeaveOneOut(matriz_datos, etiquetas, h1[0:len(h1)-1])
        h2[len(h2)-1] = funcionEvaluacionLeaveOneOut(matriz_datos, etiquetas, h1[0:len(h2)-1])

        # Competición para entrar en la poblacion
        seleccion[0] = h1
        seleccion[1] = h2
        seleccion[2] = poblacion[0]
        seleccion[3] = poblacion[1]
        seleccion = seleccion[seleccion[:, -1].argsort()]
        poblacion[0] = seleccion[len(seleccion)-1]
        poblacion[1] = seleccion[len(seleccion)-2]

        # Se ordena la poblacion
        poblacion = poblacion[poblacion[:, -1].argsort()]
        iterations += 2

    return poblacion[-1][0:len(matriz_datos[0])], poblacion[-1][tam_w]

def generacional(entrenamiento: list) -> np.array: 
    '''
    # Param: conjunto de entrenamiento (Sin particion de evaluación) 
    # Return: pesos optimizados con BLX 
    '''
    # Generación de la solucion inicial
    NUM_CARACTERISTICAS =   len(entrenamiento[0].caracteristicas)
    STDEV = 0.2 # Este valor es menor y viene indicado en las transparencias
    MEDIA = 0
    pesos = np.random.rand(NUM_CARACTERISTICAS)
 
    # Calculo de los cruces esperados 
    probabilidad_cruce = 0.6
    tam_poblacion = entrenamiento.size()
    cruces_esperados = probabilidad_cruce*tam_poblacion/2
    
    # Calculo de las mutaciones esperadas 
    n_genes = entrenamiento[0].size() 
    probabilidad_mutacion = 0.01
    mutaciones_esperadas = tam_poblacion*n_genes*probabilidad_mutacion

    return pesos

def AGG_BLX(entrenamiento): 
    return generacional(entrenamiento, cruce_BLX, 15000, 0.1, 0.7)

def AGG_CA(entrenamiento): 
    return generacional(entrenamiento, cruce_CA, 15000, 0.1, 0.7)

def AGE_BLX(entrenamiento: np.array, etiquetas: np.array) -> np.array: 
    return estacionario(entrenamiento, etiquetas, cruce_BLX, 15000, 0.1)

def AGE_CA(entrenamiento): 
    return estacionario(entrenamiento, cruce_CA, 15000, 0.1)