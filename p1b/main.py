import random 

# Lectura de los datos del archivo 
def leerDatos(nombreArchivo): 
    archivo = open(nombreArchivo, 'r')
    lineas = archivo.readlines()
    # Ignorar metadatos 
    while lineas[0] != "@data\n": 
        lineas.pop(0)
    lineas.pop(0)
    # Guardamos los datos 
    datos = []
    for line in lineas:
        # Separar cada linea en sus valores, con la coma ',' como separador
        line = line.split(',')
        for i in range(len(line) - 1):
            line[i] = float(line[i])
        datos.append(line)
    archivo.close()
    return datos

# Divisor de la muestra en 5 
def divisor(muestra):
    random.shuffle(muestra) 
    divisiones = [[],[],[],[],[]]
    quinto = len(muestra) // 5 
    for i in range(quinto): 
        divisiones[0].append(muestra[i])
    for i in range(quinto, 2*quinto): 
        divisiones[1].append(muestra[i])
    for i in range(2*quinto, 3*quinto):
        divisiones[2].append(muestra[i])
    for i in range(3*quinto, 4*quinto): 
        divisiones[3].append(muestra[i]) 
    for i in range(4*quinto, len(muestra)): 
        divisiones[4].append(muestra[i])    
    return divisiones

# Algoritmo 1-NN
def KNN (data, indice): 
    entrenamiento = data[1]
    test = data[0]
    tasa_clas = 0
    tasa_red = 0 
    tiempo = 0 
    return tasa_clas, tasa_red, tiempo 

# Algoritmo BL 

# Algoritmo greedy RELIEF 

# Main 
print('Lectura de los datos')
ionosphere = leerDatos('Instancias_APC/ionosphere.arff')
parkinsons = leerDatos('Instancias_APC/parkinsons.arff')
spectf_heart = leerDatos('Instancias_APC/spectf-heart.arff')
print('Lectura de los datos completada')

print('Algoritmo 1-NN')
divisiones = divisor(ionosphere)
# divisiones = divisor(parkinsons)
# divisiones = divisor(spectf_heart)
tasa_clas = 0 
tasa_red = 0 
tiempo = 0 
tasa_clasi = 0 
tasa_redi = 0 
tiempoi = 0 
for i in range(5): 
    tasa_clasi, tasa_red, tiempo = KNN(divisiones, i)
    tasa_clas += tasa_clasi 
    tasa_red += tasa_redi
    tiempo += tiempoi 
tasa_clas_media = tasa_clas / 5 
tasa_red_media = tasa_red / 5 
tiempo_medio = tiempo / 5 


print('Algoritmo 1-NN completado')