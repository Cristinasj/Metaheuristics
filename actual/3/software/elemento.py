import numpy as np

class Elemento: 
    def __init__(self, fila): 
        fila = list(fila)
        self.clase = fila[-1]
        self.caracteristicas = np.array(fila[:-1])

    # Return: distancia cuadrada 
    def distancia(self, otro, pesos=None):  
        if len(self.caracteristicas) != len(otro.caracteristicas): 
            raise Exception("Longitud diferente")
        if pesos is None: 
            pesos = np.full_like(otro, 1)
        return np.sum((self.caracteristicas-otro.caracteristicas)**2 * pesos)

    def amigo(self, conjunto): 
        conjunto_amigo = [x for x in conjunto if x.clase == self.clase]
        distancias = list([self.distancia(x) for x in conjunto_amigo])
        indice_amigo = np.array(distancias).argmin()
        return conjunto_amigo[indice_amigo]
    
    
    def enemigo(self, conjunto): 
        conjunto_enemigo = [x for x in conjunto if x.clase != self.clase]
        distancias = list([self.distancia(x) for x in conjunto_enemigo])
        indice_enemigo = np.array(distancias).argmin()
        return conjunto_enemigo[indice_enemigo]