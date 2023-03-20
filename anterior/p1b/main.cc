// Cristina Sánchez Justicia 
# include <vector>
# include <string> 
# include <fstream>
# include <iostream>

using namespace std; 


vector<pair<char,vector<double>>> leerDatos(string nombreArchivo) {
    vector<pair<char,vector<double>>> datos;
    ifstream archivo; 
    archivo.open(nombreArchivo); 
    string linea; 
    if (archivo) {
        // Se ignoran los metadatos 
        getline(archivo, linea); 
        while(linea != "@data")
            getline(archivo, linea); 
        // Se leen los datos 
        if (archivo)
            getline(archivo, linea);
            int elemento = 0; 
            while (archivo) {
                // Se divide la línea por las comas 
                vector<vector<char>> palabras;
                vector<char> palabra;  
                for (int i = 0; i < linea.size(); i++) {
                    if (linea[i] != ',')
                        palabra.push_back(linea[i]); 
                    else 
                        palabras.push_back(palabra); 
                }
                palabras.push_back(palabra); 
                // Se convierte el vector<vector<char>> en pair<char,vector<double>>>
                int i = palabras.size() - 1;
                datos[elemento].first = palabras[i][0];  
                for (int i = 0; i < palabras.size() - 1; i++) {
                    datos[elemento].second[i] = stoi(palabras[i].data()); 
                }
                elemento++; 
                // Se lee la siguiente línea 
                if (archivo) getline(archivo, linea); 
            }

    }
    return datos; 
}
 

int main () {
    // Lectura de los datos de entrada 
    
    vector<pair<char,vector<double>>> ionosphere = leerDatos("Instancias_APC/ionosphere.arff"); 
    vector<pair<char,vector<double>>> parkinsons = leerDatos("Instancias_APC/parkinsons.arff"); 
    vector<pair<char,vector<double>>> spectf_heart = leerDatos("Instancias_APC/spect-heart.arff"); 
    
    // Algoritmo 1-NN 

    // Algoritmo BL 

    // Algoritmo Greedy RELIEF  
    return 0; 
}