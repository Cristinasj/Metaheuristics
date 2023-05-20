import numpy as np
import time
import random
import arff 
     

def AGE_BLX(): 
  pass 

def porcentaje_clasificacion(): 
  pass

def porcentaje_reduccion(): 
  pass 

def funcion_evaluacion(): 
  pass 

def cargar_datos(nombre):

  data = arff.load(nombre)
  datos = np.zeros((len(data[0]), len(data[0][0])-1))

  #Cargamos datos
  for i in range(0, len(datos)):
    for j in range(0, len(datos[0])):
      datos[i][j] = data[0][i][j]

  #Cargamos los atributos
  clases = np.chararray(len(datos))
  n_atrib = len(datos[0])
  for i in range (0, len(datos)):
    clases[i] = data[0][i][n_atrib]

  return datos, clases

#Calcula el minimo y maximo para cada atributo de un conjunto de datos
def max_min_values(datos):
  max = np.zeros(len(datos[0]))
  min = np.zeros(len(datos[0]))

  min = np.apply_along_axis(np.amin, 0, datos)
  max = np.apply_along_axis(np.amax, 0, datos)

  return min, max

#Normaliza un set de datos dado el vector de minimos y el vector max-min
def normaliza(datos, min, dif):
  ret = (datos - min)/dif
  ret[np.isnan(ret)] = 0
  return ret

def dist_1NN(nuevo, datos, w):
  distancias = np.sum((datos-nuevo)**2*w, 1)**0.5
  return distancias

def classf_1NN(nuevo, datos, w):
  t1 = time.time()
  distancias = np.sum((datos-nuevo)**2*w, 1)**0.5
  #return np.where(distancias == np.amin(distancias))[0][0]
  s = np.argmin(distancias)
  t2 = time.time()
  return s

def classf_1NN_(nuevo, datos, w, i):
  distancias = np.sum((datos-nuevo)**2*w, 1)**0.5
  distancias[i] = np.Inf
  #return np.where(distancias == np.amin(distancias))[0][0]
  return np.argmin(distancias)

#Ejecuta el classificador 1-NN para un ejemplo de datos y de test
def alg_1NN(datos, test, w, max_values, min_values):

  dif = max_values - min_values
  test = (test - min_values)/dif
  datos = normaliza(datos, min_values, dif)

  index_clases = np.zeros(len(test))
  for i in range(0, len(test)):
    index_clases[i] = classf_1NN(test[i], datos, w)

  return index_clases

#Calcular error para un conjunto de train y test
def calcular_error(datos, clas_d, test, clas_t, w):
  error = 0.0
  min_val, max_val = max_min_values(datos)
  datos = normaliza(datos, min_val, max_val-min_val)

  min_val, max_val = max_min_values(test)
  test = normaliza(test, min_val, max_val-min_val)

  for i in range(0, len(test)):
    c = classf_1NN(test[i], datos, w)
    if (clas_d[c] == clas_t[i]):
      error += 1

  return error/len(test)

#Calcula el error sobre un conjunto (leave one out)
def calcular_error_leave(datos, clases, w):

  w_temp = w.copy()
  w_temp[w_temp < 0.1] = 0

  aciertos = 0
  for i in range(0, len(datos)):
    c = classf_1NN_(datos[i], datos, w_temp, i)
    if (clases[c] == clases[i]):
      aciertos += 1

  return aciertos/len(datos)

def evaluacion(datos, clases, w, alpha):
  error = calcular_error_leave(datos, clases, w)
  return alpha*error + (1 - alpha)*( len(w[w < 0.1])/len(w) )
     

def evaluacion_detallada(datos, clases, w, alpha):
  error = calcular_error_leave(datos, clases, w)
  return alpha*error + (1 - alpha)*( len(w[w < 0.1])/len(w) ), error, ( len(w[w < 0.1])/len(w) )

def cruce_BLX(c1, c2):

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

def cruce_aritmetico(c1, c2):

  alpha = np.random.rand(1)[0]
  salida1 = np.zeros(len(c1)+1)
  salida2 = np.zeros(len(c1)+1)

  salida1[0:len(c1)] = c1*alpha + c2*(1-alpha)
  salida2[0:len(c1)] = c1*(1-alpha) + c2*alpha

  return salida1, salida2



def mutar(i, j, poblacion):

  poblacion[i][j] += np.random.normal(0, 0.3)
  if (poblacion[i][j] < 0):
    poblacion[i][j] = 0
  if (poblacion[i][j] > 1):
    poblacion[i][j] = 1
     


# Recibe 4 individuos y selecciona 2 por torneo
def torneo4(seleccion):
  if (seleccion[0][-1] > seleccion[1][-1]):
    c1 = seleccion[0]
  else:
    c1 = seleccion[1]

  if (seleccion[2][-1] > seleccion[3][-1]):
    c2 = seleccion[2]
  else:
    c2 = seleccion[3]

  return c1[0:-1], c2[0:-1]

# Recibe 2 individuos y selecciona el mejor por torneo
def torneo2(seleccion, poblacion):

  c1 = poblacion[seleccion[0]]
  c2 = poblacion[seleccion[1]]

  if (c1[-1] > c2[-1]):
    s = seleccion[0]
  else:
    s = s = seleccion[1]

  return s
     


def estacionario(datos, label, cruce, tam_pob, iteraciones_max, prob_mut):
  
  tam_w = len(datos[0])
  n_genes = tam_pob * tam_w
  
  #Creo aleatoriamente la poblacion inicial
  poblacion = np.zeros((tam_pob, len(datos[0])+1))
  for i in range(0, tam_pob):
    w = np.random.rand(len(datos[0]))
    poblacion[i][0:len(datos[0])] = w
    poblacion[i][len(datos[0])] = evaluacion(datos, label, w, 0.5)

  poblacion = np.array(poblacion)
  poblacion = poblacion[poblacion[:, -1].argsort()]

  iterations = 0
  while (iterations < iteraciones_max):

    # Seleccionamos 4 individuos y elegimos 2 por torneo
    seleccion = poblacion[random.sample(range(tam_pob), 4)]
    c1, c2 = torneo4(seleccion)

    # Hacemos 2 cruces
    seleccion = np.zeros((4, len(datos[0])+1))
    h1, h2 = cruce(c1, c2)

    # Aplicamos las mutaciones
    mutaciones = [random.sample(range(tam_w*2), (int)(2*tam_w*prob_mut))]
    for i in range(0, len(mutaciones)):
      r = np.random.randint(tam_w*2)
      i = r // tam_w
      j = r % tam_w
      mutar(i, j, [h1, h2])
    
    h1[len(h1)-1] = evaluacion(datos, label, h1[0:len(h1)-1], 0.5)
    h2[len(h2)-1] = evaluacion(datos, label, h1[0:len(h2)-1], 0.5)

    # Los hijos compiten para entrar en la poblacion
    seleccion[0] = h1
    seleccion[1] = h2
    seleccion[2] = poblacion[0]
    seleccion[3] = poblacion[1]
    seleccion = seleccion[seleccion[:, -1].argsort()]
    poblacion[0] = seleccion[len(seleccion)-1]
    poblacion[1] = seleccion[len(seleccion)-2]

    # Ordenamos la poblacion
    poblacion = poblacion[poblacion[:, -1].argsort()]

    iterations += 2

  return poblacion[-1][0:len(datos[0])], poblacion[-1][tam_w]
     
def generacional(datos, label, cruce, tam_pob, iteraciones_max, prob_cruce, prob_mut):

  tam_w = len(datos[0])
  n_genes = tam_pob * tam_w
  
  #Creo aleatoriamente la poblacion inicial
  poblacion = np.zeros((tam_pob, len(datos[0])+1))
  for i in range(0, tam_pob):
    w = np.random.rand(len(datos[0]))
    poblacion[i][0:len(datos[0])] = w
    poblacion[i][len(datos[0])] = evaluacion(datos, label, w, 0.5)
  
  iterations = 0
  while (iterations < iteraciones_max):

    mejor_individuo = poblacion[np.argmax(poblacion[:, -1])]

    # Seleccionamos 4 individuos y elegimos 2 por torneo
    seleccion = np.zeros(tam_pob, dtype=np.intc)
    for i in range(0, tam_pob):
      r1 = np.random.randint(tam_pob)
      r2 = np.random.randint(tam_pob)
      seleccion[i] = torneo2([r1, r2], poblacion)

    cruzar = poblacion[seleccion[0:(int)(tam_pob*prob_cruce)+1]]
    #cruzar = np.delete(cruzar, -1, 1)
    no_cruce = poblacion[seleccion[(int)(tam_pob*prob_cruce)+1:tam_pob]]

    # Realizamos los cruces y creamos la nueva generacion

    nueva_generacion = np.zeros((len(cruzar), tam_w+1))
    for i in range(0, len(cruzar)//2):
        a, b = cruce(cruzar[i][0:tam_w], cruzar[i+len(cruzar)//2][0:tam_w])
        nueva_generacion[i] = a
        nueva_generacion[i+len(cruzar)//2] = b
    nueva_generacion = np.concatenate((nueva_generacion, no_cruce), axis=0)

    # Aplicamos las mutaciones
    evaluar = np.arange((int)(tam_pob*prob_cruce)+1)
    evaluar_mut = []
    for n in range(0, (int)(tam_pob*prob_mut)):
      r = np.random.randint(n_genes)
      i = r // tam_w
      j = r % tam_w
      mutar(i, j, nueva_generacion)
      if (i >= (int)(tam_pob*prob_cruce)+1):
        evaluar_mut.append(i)
    evaluar_mut = np.unique(evaluar_mut)
    
    #Calculamos el fitness
    for i in range(0, len(evaluar)):
      w = nueva_generacion[evaluar[i]]
      w[len(w)-1] = evaluacion(datos, label, w[0:len(w)-1], 0.5)
    for i in range(0, len(evaluar_mut)):
      w = nueva_generacion[evaluar_mut[i]]
      w[len(w)-1] = evaluacion(datos, label, w[0:len(w)-1], 0.5)

    peor_individuo = np.argmin(poblacion[:, -1])
    if (mejor_individuo[-1] > nueva_generacion[peor_individuo][-1]):
      nueva_generacion[peor_individuo] = mejor_individuo

    poblacion = nueva_generacion

    iterations += len(evaluar)+len(evaluar_mut)

  return poblacion[np.argmax(poblacion[:, -1])][0:tam_w], poblacion[np.argmax(poblacion[:, -1])][tam_w]



#Obtener vecino de w
def operador_mutacion(w, i):
  w_nuevo = w
  w_nuevo[i] += np.random.normal(0, 0.3)
  if (w_nuevo[i] < 0):
    w_nuevo[i] = 0
  if (w_nuevo[i] > 1):
    w_nuevo[i] = 1

  return w_nuevo

def BL_search(datos, clases, w_ini):

  #Normalizar e inicializar w
  w_mejor = w_ini.copy()
  w = w_mejor.copy()

  contador_greedy = 0
  i_w = 0
  iterations = 0
  fitness_actual = 0.0
  indx_w = np.arange(0, len(w))
  np.random.shuffle(indx_w)

  while(contador_greedy < len(w)*2):

    #Se calcula el fitness para w
    #aciertos = calcular_error_leave(datos, clases, w)
    fitness = evaluacion(datos, clases, w, 0.5)

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
    w = operador_mutacion(w, indx_w[i_w])
    iterations += 1
    i_w = (i_w+1)%len(w)

  return w, fitness_actual
     


def memeticos(datos, label, cruce, tam_pob, iteraciones_max, prob_cruce, prob_mut, n_bl, bl_mejores=False):

  tam_w = len(datos[0])
  n_genes = tam_pob * tam_w
  
  #Creo aleatoriamente la poblacion inicial
  poblacion = np.zeros((tam_pob, len(datos[0])+1))
  for i in range(0, tam_pob):
    w = np.random.rand(len(datos[0]))
    poblacion[i][0:len(datos[0])] = w
    poblacion[i][len(datos[0])] = evaluacion(datos, label, w, 0.5)
  
  iterations = 0
  meme_count = 0
  while (iterations < iteraciones_max):

    # Cada 10 generaciones hacemos BL 
    if (meme_count == 10):
      meme_count = 0

      if (bl_mejores):
        poblacion = poblacion[poblacion[:, -1].argsort()]
      # Si no, se hace seleccion aleatoria
      else:
        np.random.shuffle(poblacion)
        
      # Aplicamos BL
      for i in range(0,(int)(n_bl*tam_pob)):
        w, fitness = BL_search(datos, label, poblacion[i][0:tam_w])
        poblacion[i][0:tam_w] = w
        poblacion[i][tam_w] = fitness

    mejor_individuo = poblacion[np.argmax(poblacion[:, -1])]

    # Seleccionamos 4 individuos y elegimos 2 por torneo
    seleccion = np.zeros(tam_pob, dtype=np.intc)
    for i in range(0, tam_pob):
      r1 = np.random.randint(tam_pob)
      r2 = np.random.randint(tam_pob)
      seleccion[i] = torneo2([r1, r2], poblacion)

    cruzar = poblacion[seleccion[0:(int)(tam_pob*prob_cruce)+1]]
    #cruzar = np.delete(cruzar, -1, 1)
    no_cruce = poblacion[seleccion[(int)(tam_pob*prob_cruce)+1:tam_pob]]

    # Realizamos los cruces y creamos la nueva generacion

    nueva_generacion = np.zeros((len(cruzar), tam_w+1))
    for i in range(0, len(cruzar)//2):
        a, b = cruce(cruzar[i][0:tam_w], cruzar[i+len(cruzar)//2][0:tam_w])
        nueva_generacion[i] = a
        nueva_generacion[i+len(cruzar)//2] = b
    nueva_generacion = np.concatenate((nueva_generacion, no_cruce), axis=0)

    # Aplicamos las mutaciones
    evaluar = np.arange((int)(tam_pob*prob_cruce)+1)
    evaluar_mut = []
    for n in range(0, (int)(tam_pob*prob_mut)):
      r = np.random.randint(n_genes)
      i = r // tam_w
      j = r % tam_w
      mutar(i, j, nueva_generacion)
      if (i >= (int)(tam_pob*prob_cruce)+1):
        evaluar_mut.append(i)
    evaluar_mut = np.unique(evaluar_mut)
    
    #Calculamos el fitness
    for i in range(0, len(evaluar)):
      w = nueva_generacion[evaluar[i]]
      w[len(w)-1] = evaluacion(datos, label, w[0:len(w)-1], 0.5)
    for i in range(0, len(evaluar_mut)):
      w = nueva_generacion[evaluar_mut[i]]
      w[len(w)-1] = evaluacion(datos, label, w[0:len(w)-1], 0.5)

    # lo sustituimos
    peor_individuo = np.argmin(poblacion[:, -1])
    if (mejor_individuo[-1] > nueva_generacion[peor_individuo][-1]):
      nueva_generacion[peor_individuo] = mejor_individuo

    poblacion = nueva_generacion

    iterations += len(evaluar) + len(evaluar_mut) + (int)(n_bl*tam_pob)

  return poblacion[np.argmax(poblacion[:, -1])][0:tam_w], poblacion[np.argmax(poblacion[:, -1])][tam_w]
     


def cross_validation_generacional(datos, clases, cruce, tam_pob, iteraciones_max, prob_cruce, prob_mut):
  kf = KFold(n_splits=5)
  kf.get_n_splits(datos)
  KFold(n_splits=5, random_state=None, shuffle=False)
  for train_index, test_index in kf.split(datos):
    t1 = time.time()
    # generacional(datos, label, cruce, tam_pob, iteraciones_max, prob_cruce, prob_mut):
    w, error = generacional(datos[train_index], clases[train_index], cruce, tam_pob, iteraciones_max, prob_cruce, prob_mut)
    t2 = time.time()
    print("Fitness en training: ", np.around(error, 3))
    error = np.around(calcular_error(datos[train_index], clases[train_index], datos[test_index], clases[test_index], w), 3)
    red = np.around(len(w[w < 0.1])/len(w), 3)
    print("Aciertos test: ", error, " Red en test: ", red, " Fitness test: ", error*0.5+red*0.5)
    print("T: ", t2-t1)
    print("---------------")

def cross_validation_estacionario(datos, clases, cruce, tam_pob, iteraciones_max, prob_mut):
  kf = KFold(n_splits=5)
  kf.get_n_splits(datos)
  KFold(n_splits=5, random_state=None, shuffle=False)
  for train_index, test_index in kf.split(datos):
    t1 = time.time()
    # generacional(datos, label, cruce, tam_pob, iteraciones_max, prob_cruce, prob_mut):
    w, error = estacionario(datos[train_index], clases[train_index], cruce, tam_pob, iteraciones_max, prob_mut)
    t2 = time.time()
    print("Fitness en training: ", np.around(error, 3))
    error = np.around(calcular_error(datos[train_index], clases[train_index], datos[test_index], clases[test_index], w), 3)
    red = np.around(len(w[w < 0.1])/len(w), 3)
    print("Aciertos test: ", error, " Red en test: ", red, " Fitness test: ", error*0.5+red*0.5)
    print("T: ", t2-t1)
    print("---------------")

def cross_validation_memetico(datos, clases, cruce, tam_pob, iteraciones_max, prob_cruce, prob_mut, n_bl, bl_mejores):
  kf = KFold(n_splits=5)
  kf.get_n_splits(datos)
  KFold(n_splits=5, random_state=None, shuffle=False)
  for train_index, test_index in kf.split(datos):
    t1 = time.time()
    #                   (datos, label, cruce, tam_pob, iteraciones_max, prob_cruce, prob_mut, n_bl, bl_mejores=False)
    w, error = memeticos(datos[train_index], clases[train_index], cruce, tam_pob, iteraciones_max, prob_cruce, prob_mut, n_bl, bl_mejores)
    t2 = time.time()
    print("Fitness en training: ", np.around(error, 3))
    error = np.around(calcular_error(datos[train_index], clases[train_index], datos[test_index], clases[test_index], w), 3)
    red = np.around(len(w[w < 0.1])/len(w), 3)
    print("Aciertos test: ", error, " Red en test: ", red, " Fitness test: ", error*0.5+red*0.5)
    print("T: ", t2-t1)
    print("---------------")
     

datos, clases = cargar_datos('')
min, max = max_min_values(datos)
datos = normaliza(datos, min, max)
print("GENETICO ARITMETICO ++++++++++++ IONOSPHERE")
print(cross_validation_generacional(datos, clases, cruce_aritmetico, 30, 15000, 0.7, 0.1))

datos, clases = cargar_datos('/content/drive/MyDrive/MH/Instancias_APC/parkinsons.arff')
min, max = max_min_values(datos)
datos = normaliza(datos, min, max)
print("GENETICO ARITMETICO ++++++++++++ PARKINSONS")
print(cross_validation_generacional(datos, clases, cruce_aritmetico, 30, 15000, 0.7, 0.1))

datos, clases = cargar_datos('/content/drive/MyDrive/MH/Instancias_APC/spectf-heart.arff')
min, max = max_min_values(datos)
datos = normaliza(datos, min, max)
print("GENETICO ARITMETICO ++++++++++++ SPECTF-HEART")
print(cross_validation_generacional(datos, clases, cruce_aritmetico, 30, 15000, 0.7, 0.1))

datos, clases = cargar_datos('/content/drive/MyDrive/MH/Instancias_APC/ionosphere.arff')
min, max = max_min_values(datos)
datos = normaliza(datos, min, max)
print("GENETICO BLX ++++++++++++ IONOSPHERE")
print(cross_validation_generacional(datos, clases, cruce_BLX, 30, 15000, 0.7, 0.1))

datos, clases = cargar_datos('/content/drive/MyDrive/MH/Instancias_APC/parkinsons.arff')
min, max = max_min_values(datos)
datos = normaliza(datos, min, max)
print("GENETICO BLX ++++++++++++ PARKINSONS")
print(cross_validation_generacional(datos, clases, cruce_BLX, 30, 15000, 0.7, 0.1))

datos, clases = cargar_datos('/content/drive/MyDrive/MH/Instancias_APC/spectf-heart.arff')
min, max = max_min_values(datos)
datos = normaliza(datos, min, max)
print("GENETICO BLX ++++++++++++ SPECTF-HEART")
print(cross_validation_generacional(datos, clases, cruce_BLX, 30, 15000, 0.7, 0.1))
     
