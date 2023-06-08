import numpy as np 
from algoritmos.evaluacionMatriz import funcionEvaluacionLeaveOneOut
from numpy import random 

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

def ES(entrenamiento, etiquetas): 
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
  '''
  Enfriamiento simulado 
  '''
  evaluaciones_efectuadas = 0
  max_evaluaciones = 15000

  k = 1.3806e-23
  w = np.random.rand(len(entrenamiento[0]))
  f = funcionEvaluacionLeaveOneOut(entrenamiento, etiquetas, w)
  t0 = (0.3*f) / -np.log(0.3)
  tf = 10e-3
  t = t0

  max_vecinos = 10*len(w)
  max_exitos = 0.1*max_vecinos

  M = (max_evaluaciones / max_vecinos)

  B = (t0 - tf) / (M * t0 * tf)
  w_mejor = w.copy()
  f_mejor = f

  exitos = 1
  
  vecinos = 0
  exitos = 0

  while (vecinos < max_vecinos and exitos < max_exitos and evaluaciones_efectuadas < max_evaluaciones and exitos > 0):

    # Se aplica el operador de Vecino. Se escoge
    # aleatoriamente la característica i a la que se
    # aplica la perturbación
    w_ = mutacion(w, np.random.randint(len(w)))
    f_ = funcionEvaluacionLeaveOneOut(entrenamiento, etiquetas, w_)
    evaluaciones_efectuadas += 1
    vecinos += 1
    dif = f - f_

    if (dif < 0 or random.random() < np.exp((-dif)/(t*k)) ):
      
      exitos += 1
      w = w_.copy()
      f = f_
      if (f > f_mejor):
        f_mejor = f
        w_mejor = w.copy()

    t = t / (1+(B*t))
    
  return w_mejor

def ILS(entrenamiento, etiquetas):
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
  w_mejor = BL(entrenamiento, etiquetas, np.random.rand(len(entrenamiento[0])))
  f_mejor =  funcionEvaluacionLeaveOneOut(entrenamiento, etiquetas, w_mejor) 

  for i in range(num_iteraciones):

    w_mutada = mutar(w_mejor)

    w_actual = BL(entrenamiento, etiquetas, w_mutada)
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
  return np.random(len(entrenamiento[0]))

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