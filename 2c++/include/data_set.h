#ifndef DATA_SET_H_DEFINED
#define DATA_SET_H_DEFINED

#include <string>
#include <algorithm>
#include <vector>
#include "data_point.h"

class DataSet {
    private:
    std::vector<DataPoint> points;
    
    public:
    DataSet() {}
    DataSet(std::vector<DataPoint> vdp) { points = vdp; }

    /**
     * @brief Extract dataset from arff file
     */
    DataSet(std::string arffFileName);

    std::vector<DataPoint>& getDataPoints() {return points;};

    /**
     * Split dataset into k equal portions, and return copies.
     * @return std::vector<DataSet> 
     */
    std::vector<DataSet> split(int k, bool shuffle=true);

    /**
     * @brief Merge the DataPoints in another set into this one.
     * @return DataSet 
     */
    void merge(DataSet other);
    
    /**
     * @brief Merge a vector of DataSets into this one.
     * @return DataSet 
     */
    void merge(std::vector<DataSet> vec);

    /**
     * @brief Normaliza todo a valores en [0,1], conservando la 
     * proporcionalidad entre los elementos de una columna.
     * @pre Todos los DataPoint contenidos tienen el mismo número de atributos.
     */
    void normalizarColumnas();

    friend std::istream& operator>>(std::istream&, DataSet&);
    friend std::ostream& operator<<(std::ostream&, DataSet&);

};

#endif