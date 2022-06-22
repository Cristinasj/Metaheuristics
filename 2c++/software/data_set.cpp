#include <unistd.h>
#include "data_set.h"
#include "random.hpp"

using Random=effolkronium::random_static;

DataSet::DataSet(std::string arffFileName) {
    std::ifstream ifs(arffFileName);
    if (!ifs) {
        throw std::invalid_argument("That file could not be opened.");
    }
    std::cout << "Reading " << arffFileName << std::endl;
    ifs >> *this;
}



// Split dataset into k of (almost) equal size
std::vector<DataSet> DataSet::split(int k, bool shuffle) {
    DataSet copy = *this;
    // Shuffle 
    if (shuffle) {
        Random::shuffle(copy.points);
    }

    std::vector<DataSet> data_sets;
    int length = copy.points.size() / k;
    for (int id_ds=0; id_ds<k; ++id_ds) {
        int start_idx = id_ds * length;
        int end_idx = (id_ds+1) * length;
        auto first_it = copy.points.begin() + start_idx;
        auto last_it = copy.points.begin() + end_idx;
        data_sets.push_back(DataSet(std::vector<DataPoint>(first_it, last_it)));
    }
    return data_sets;
}

void DataSet::merge(DataSet other) {
    for (DataPoint dp : other.points) {
        this->points.push_back(dp);
    }
}

void DataSet::merge(std::vector<DataSet> vec) {
    for (DataSet ds : vec) {
        this->merge(ds);
    }
}


void clearSpaces(std::string& s);

std::istream& operator>>(std::istream& is, DataSet& ds) {
    // Getline until it's @data
    std::string line = "";
    while (line != "@data") {
        std::getline(is, line);
        clearSpaces(line);
    }

    // Then while there is something in the stream, get datapoint.
    while (is.peek() != -1) {
        DataPoint dp; is >> dp;
        ds.points.push_back(dp);
    }

    return is;
}



std::ostream& operator<<(std::ostream& os, DataSet& ds) {

    os << "DATASET (" << ds.points.size() << ")" << std::endl;
    for (DataPoint p : ds.points) {
        os << "\t" << p << std::endl;
    }

    return os;
}


void clearSpaces(std::string& s) {
    while (!s.empty() && isspace(s.at(0))) {
        s.erase(0, 1);
    }
    while (!s.empty() && isspace(s.at(s.size()-1))) {
        s.erase(s.size()-1, 1);
    }
}


void DataSet::normalizarColumnas() {
    if (this->getDataPoints().empty()) {
        return;
    }
    int num_cols = this->getDataPoints()[0].getNumAttributes();
    int num_rows = this->getDataPoints().size();
    for (int col=0; col<num_cols; col++) {
        // find min and max
        double min_val = 999999;
        double max_val = -999999;
        for (int row=0; row<num_rows; ++row) {
            double current_val = this->getDataPoints()[row].getAttributes()[col];
            if (min_val > current_val) {
                min_val = current_val;
            }
            if (current_val > max_val) {
                max_val = current_val;
            }
        }

        // Normalize each value in the current column to [0,1]
        for (int row=0; row<num_rows; row++) {
            points[row].getAttributes()[col] -= min_val;
            if (max_val - min_val != 0) {
                points[row].getAttributes()[col] /= (max_val - min_val);
            }
        }
    }
}