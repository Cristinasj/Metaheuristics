import numpy as np

class Elemento: 
    def __init__(self, fila): 
        fila = list(fila)
        self.clase = fila[-1]
        self.caracteristicas = np.array(fila[:-1])

    def distancia(self, otro, pesos=None):  
        if len(self.caracteristicas) != len(otro.caracteristicas): 
            raise Exception("Longitud diferente")
        if pesos == None: 
            pesos = np.full_like(otro, 1)
        return np.sqrt(np.sum((self.caracteristicas-otro.caracteristicas)**2) * pesos)
