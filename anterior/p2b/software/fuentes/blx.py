import numpy as np

# Parámetros: 
#       - madre, padre: dos vectores de pesos  
# Return: 
#       - list con dos vectores de pesos descendientes 
def blx(madre, padre): 
    ALPHA = 0.3
    # Vectores con Cmin y Cmax correspondientes para cada peso 
    min = [] 
    max = []
    for i in range(madre.shape):
        minimo = np.minimun(madre[i], padre[i])
        maximo = np.maximun(madre[i], padre[i])
        I = maximo - minimo 
        min.append(minimo - I*ALPHA)
        max.append(maximo + I*ALPHA)
    devolver1 = []
    devolver2 = []  
    for i in range(min):
        devolver1.append(np.random.uniform(min[i],max[i])) 
        devolver2.append(np.random.uniform(min[i],max[i])) 
    return devolver1, devolver2 

