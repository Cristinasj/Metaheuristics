# Return: clase del vecino más cercano 
def unoNN(entrenamiento, elemento, pesos=None): 
    distancias = list([elemento.distancia(e, pesos) for e in entrenamiento])
    minimo = np.array(distancias).argmin()
    return entrenamiento[minimo].clase

