import numpy as np 
from algoritmos.evaluacionMatriz import funcionEvaluacionLeaveOneOut
from numpy import random 
import math

def mutacion(w, i):
  '''
  Obtener vecino de w
  '''
  w_nuevo = w.copy()
  w_nuevo[i] += np.random.normal(0, 0.3)
  if (w_nuevo[i] < 0):
    w_nuevo[i] = 0
  if (w_nuevo[i] > 1):
    w_nuevo[i] = 1

  return w_nuevo

def BL(entrenamiento, etiquetas, w_ini):
  '''
  parametros: 

    entrenamiento: Matriz con el conjunto de 
    entrenamiento. Las filas son los elementos y 
    las columnas son el valor para la caracteristica 
    coreespondiente

    etiquetas: Clase a la que pertenece cada fila de la matriz 

    w_ini: Vector de inicio para la búsqueda local 

  return: 
    pesos optimizados con la Búsqueda Local  
  '''
  max_iter = 1000
    
  #Normalizar e inicializar w
  w_mejor = w_ini.copy()
  w = w_mejor.copy()

  contador_greedy = 0
  i_w = 0
  iterations = 0
  fitness_actual = 0.0
  indx_w = np.arange(0, len(w))
  np.random.shuffle(indx_w)

  while(contador_greedy < len(w)*10 and iterations < max_iter):

    #Se calcula el fitness para w
    fitness = funcionEvaluacionLeaveOneOut(entrenamiento, etiquetas, w)

    #Actualizar fitness y w si procede
    if (fitness > fitness_actual):
      fitness_actual = fitness
      w_mejor = w
      contador_greedy = 0
      i_w = 0
      np.random.shuffle(indx_w)
    else:
      contador_greedy += 1

    #Se calcula nuevo w + control de indices
    w = mutacion(w, indx_w[i_w])
    iterations += 1
    i_w = (i_w+1)%len(w)

  return w

def BMB(entrenamiento, etiquetas):  
  '''
  parametros: 

    entrenamiento: Matriz con el conjunto de 
    entrenamiento. Las filas son los elementos y 
    las columnas son el valor para la caracteristica 
    coreespondiente

    etiquetas: Clase a la que pertenece cada fila de la matriz 

  return: 
    pesos optimizados con la Búsqueda Local  
  '''
  num_arranques = 15

  # Ejecución inicial para comparar 
  w_ini = np.random.rand(len(entrenamiento[0]))
  w_mejor = BL(entrenamiento, etiquetas, w_ini)
  f_mejor = funcionEvaluacionLeaveOneOut(entrenamiento, etiquetas, w_mejor)
  
  for i in range(num_arranques):
    w = BL(entrenamiento, etiquetas, np.random.rand(len(entrenamiento[0])))
    f = funcionEvaluacionLeaveOneOut(entrenamiento, etiquetas, w)
    if (f > f_mejor):
      f_mejor = f
      w_mejor = w.copy()
  return w_mejor

def ES(entrenamiento, etiqueta):  
  '''
  Algoritmo de enfriamiento simulado para resolver el problema. Los pesos de inicio son aleatorios.  
  parametros: 

    entrenamiento: Matriz con el conjunto de 
    entrenamiento. Las filas son los elementos y 
    las columnas son el valor para la caracteristica 
    coreespondiente

    etiquetas: Clase a la que pertenece cada fila de la matriz 

  return: 
    pesos optimizados con la Búsqueda Local  
  '''
  return ES_basico(entrenamiento, etiqueta, np.random.rand(len(entrenamiento[0])))

def ES_basico(entrenamiento, etiquetas, w): 
  '''
  Algoritmo de enfriamiento simulado que recibe como parámetro los pesos con los que comenzar 
  parametros: 

    entrenamiento: Matriz con el conjunto de 
    entrenamiento. Las filas son los elementos y 
    las columnas son el valor para la caracteristica 
    coreespondiente

    etiquetas: Clase a la que pertenece cada fila de la matriz 

    w: Pesos con los que comenzar    
  return: 
    pesos optimizados con la Búsqueda Local  
  '''
  u = 0.3 
  mejor_error = funcionEvaluacionLeaveOneOut(entrenamiento, etiquetas, w)
  CS0 = mejor_error
  phi = 0.2
  neg_ln_phi = -math.log(phi)
  temperatura_inicial=u*CS0/(neg_ln_phi) 
  factor_enfriamiento=0.95 
  iteraciones_temperatura=100
  iteraciones_por_vecino=15
  # Inicializar el vector de pesos
  mejor_w = w.copy()
  
  # Bucle principal de enfriamiento simulado
  temperatura = temperatura_inicial
  for i in range(iteraciones_temperatura):
      for j in range(iteraciones_por_vecino):
          # Generar un vecino aleatorio
          vecino = mejor_w + np.random.normal(0, 0.1, size=w.shape)
          
          # Calcular el error del vecino
          error_vecino = funcionEvaluacionLeaveOneOut(entrenamiento, etiquetas, vecino)
          
          # Determinar si se acepta el vecino como la mejor solución
          if error_vecino < mejor_error:
              mejor_w = vecino
              mejor_error = error_vecino
          else:
              probabilidad_aceptacion = np.exp(-(error_vecino - mejor_error) / temperatura)
              if np.random.rand() < probabilidad_aceptacion:
                  mejor_w = vecino
                  mejor_error = error_vecino
      
      # Enfriar la temperatura
      temperatura *= factor_enfriamiento
  
  return mejor_w

def ILS(entrenamiento, etiquetas): 
  '''
  Entrenador ILS que es llamado desde main

  parametros: 

    entrenamiento: Matriz con el conjunto de 
    entrenamiento. Las filas son los elementos y 
    las columnas son el valor para la caracteristica 
    coreespondiente

    etiquetas: Clase a la que pertenece cada fila de la matriz 

  return: 
    pesos optimizados con la Búsqueda Local  
  '''
  return ILS_basico(entrenamiento, etiquetas, BL)

def ILS_basico(entrenamiento, etiquetas, funcion_busqueda):
  '''
  ILS para generar el entrenador ILS que se usa en main o ILS-ES, 
  dependiendo de si se le pasa BL o ES como función de búsqueda

  parametros: 

    entrenamiento: Matriz con el conjunto de 
    entrenamiento. Las filas son los elementos y 
    las columnas son el valor para la caracteristica 
    coreespondiente

    etiquetas: Clase a la que pertenece cada fila de la matriz 

    funcion_busqueda: Puede ser ES o BL 

  return: 
    pesos optimizados con la Búsqueda Local  
  '''
  def mutar(w):
    w_mut = w.copy()
    index = [i for i in range(len(w)) if w[i] > 0.4]
    np.random.shuffle(index)
    t = (int)(len(index)*0.1)
    for i in range(0, t):
      w_mut[i] += np.random.normal(0, 0.3)
      if (w_mut[i] < 0):
        w_mut[i] = 0
      if (w_mut[i] > 1):
        w_mut[i] = 1

    return w_mut

  num_iteraciones = 15

  # Primera solución inicial aleatoria
  w_mejor = funcion_busqueda(entrenamiento, etiquetas, np.random.rand(len(entrenamiento[0])))
  f_mejor =  funcionEvaluacionLeaveOneOut(entrenamiento, etiquetas, w_mejor) 

  for i in range(num_iteraciones):

    w_mutada = mutar(w_mejor)

    w_actual = funcion_busqueda(entrenamiento, etiquetas, w_mutada)
    f_actual = funcionEvaluacionLeaveOneOut(entrenamiento, etiquetas, w_actual)
    if (f_actual > f_mejor):
      f_mejor = f_actual
      w_mejor = w_actual

  return w_mejor
    
def ILS_ES(entrenamiento, etiquetas): 
  '''
  parametros: 

    entrenamiento: Matriz con el conjunto de 
    entrenamiento. Las filas son los elementos y 
    las columnas son el valor para la caracteristica 
    coreespondiente

    etiquetas: Clase a la que pertenece cada fila de la matriz 

  return: 
    pesos optimizados con la Búsqueda Local  
  '''
  return ILS_basico(entrenamiento, etiquetas, ES_basico)

def VLS(entrenamiento, etiquetas): 
  '''
  parametros: 

    entrenamiento: Matriz con el conjunto de 
    entrenamiento. Las filas son los elementos y 
    las columnas son el valor para la caracteristica 
    coreespondiente

    etiquetas: Clase a la que pertenece cada fila de la matriz 

  return: 
    pesos optimizados con la Búsqueda Local  
  '''
  return np.random(len(entrenamiento[0]))