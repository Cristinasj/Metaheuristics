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
def BL(entrenamiento, etiquetas):
  w_ini = np.random.rand(len(entrenamiento[0]))
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
  num_arranques = 0
  w_mejor = BL(entrenamiento, etiquetas)
  for i in range(1, num_arranques):
    w = BL(entrenamiento, etiquetas, np.random.rand(len(entrenamiento[0])), 1000)
    f = funcionEvaluacionLeaveOneOut(entrenamiento, etiquetas, w, 0.5)
    if (f > f_mejor):
      f_mejor = f
      w_mejor = w.copy()
  return w_mejor

def ES(entrenamiento, etiquetas): 
  evaluaciones = 0
  max_evaluaciones = 100000

  k = 1.3806e-23
  w = np.random.rand(len(entrenamiento[0]))
  f = funcionEvaluacionLeaveOneOut(entrenamiento, etiquetas, w, 0.5)
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
  while (evaluaciones < max_evaluaciones and exitos > 0):
    vecinos = 0
    exitos = 0

    while (vecinos < max_vecinos and exitos < max_exitos and evaluaciones < max_evaluaciones):

      w_ = mutacion(w, np.random.randint(len(w)))
      f_ = funcionEvaluacionLeaveOneOut(entrenamiento, etiquetas, w_, 0.5)
      evaluaciones += 1
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
  num_iteraciones = 10000

  w, f = BL(entrenamiento, etiquetas, np.random.rand(len(entrenamiento[0])), 1000)
  w_mejor = w.copy()
  f_mejor = f 

  for i in range(1, num_iteraciones):

    w_2 = mutacion(w)
    f_2 = funcionEvaluacionLeaveOneOut(entrenamiento, etiquetas, w_2, 0.5)

    w_3, f_3 = BL(entrenamiento, etiquetas, w_2, 1000)

    # f_3 siempre será igual o mejor que f_2, no hace falta compararlos
    if (f_3 > f):
      f = f_3
      w = w_3

    # Actualizamos la mejor solucion
    if (f > f_mejor):
      f_mejor = f
      w_mejor = w.copy()

  return w_mejor, f_mejor
    
def ILS_ES(entrenamiento, etiquetas): 
    return np.random(len(entrenamiento[0]))

def VNS(entrenamiento, etiquetas): 
    return np.random(len(entrenamiento[0]))