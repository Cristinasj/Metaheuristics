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

# Algoritmo 1-NN

# Algoritmo BL 

# Algoritmo greedy RELIEF 

# Main 
print('Lectura de los datos')
ionosphere = leerDatos('Instancias_APC/ionosphere.arff')
parkinsons = leerDatos('Instancias_APC/parkinsons.arff')
spectf_heart = leerDatos('Instancias_APC/spectf-heart.arff')
print('Lectura de los datos completada')
print("Ionosphere")
print(ionosphere)
print('Algoritmo 1-NN')

print('Algoritmo 1-NN completado')