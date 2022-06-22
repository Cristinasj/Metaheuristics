#include "data_point.h"


double DataPoint::getDistanceTo(DataPoint& otro, std::vector<double>& weights) {
    if (this->getNumAttributes() != otro.getNumAttributes()) {
        throw std::runtime_error("DataPoint#getDistanceTo: Attempting to compare two data points with non-matching lengths.");
    }

    double dist_sum = 0;
    for (int i=0; i<otro.getNumAttributes(); ++i) {
        double diff = fabs(this->attributes[i] - otro.attributes[i]);
        dist_sum += weights[i] * diff*diff;
    }
    return dist_sum;
}




/**
 * Override the >> operator to read in a point (line)
 */
std::istream& operator>> (std::istream& is, DataPoint& dp) {

    // Read line
    std::string line;
    std::getline(is, line);

    // Split the line on commas
    std::vector<std::string> fields;
    while (line.size()>0) {
        int posColon = line.find(',');
        if (posColon < 0) {
            fields.push_back(line);
            break;
        }
        std::string w = line.substr(0, posColon);
        fields.push_back(w);
        line.erase(0, posColon+1);
    }

    // For each attribute except until size()-1, add 
    // to attributes. 
    for (int i=0; i<fields.size()-1; ++i) {
        std::istringstream iss(fields[i]);
        double field; 
        iss >> field;
        dp.attributes.push_back(field);
    }

    // The last one is the label.
    std::istringstream iss(fields[fields.size()-1]);
    iss >> dp.label;

    return is;
}


std::ostream& operator<< (std::ostream& os, DataPoint& dp) {
    os << "(Attributes: [";
    for (int i=0; i<dp.attributes.size(); i++) {
        auto attribute = dp.attributes[i];
        if (i == dp.attributes.size()-1) {
            os << attribute;
        }
        else {
            os << attribute << ", ";
        }

    }
    os << "], label: " << dp.label << "]";

    return os;
}