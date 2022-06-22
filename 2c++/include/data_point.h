#ifndef DATA_POINT_H_DEFINED
#define DATA_POINT_H_DEFINED

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <cmath>

/**
 * @brief Data point 
 * 
 */
class DataPoint {
private: 
    std::vector<double> attributes;
    std::string label;

public: 
    DataPoint() {}
    DataPoint(std::vector<double> atr, std::string l): label(l) { attributes = atr; }

    int getNumAttributes() {
        return attributes.size();
    }
    std::vector<double>& getAttributes() { return attributes; }

    /**
     * @brief Get weighted uclidean distance to another datapoint.
     * 
     * @return double 
     */
    double getDistanceTo(DataPoint& otro, std::vector<double>& weights);

    std::string getLabel() { return label; }

    friend std::istream& operator>> (std::istream&, DataPoint&);
    friend std::ostream& operator<< (std::ostream&, DataPoint&);
};

#endif