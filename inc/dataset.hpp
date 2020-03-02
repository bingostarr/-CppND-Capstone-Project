/*
 * dataset.hpp
 *
 *  Created on: Feb 13, 2020
 *      Author: bingo
 */

#ifndef INC_DATASET_HPP_
#define INC_DATASET_HPP_

#include <cassert>
#include <vector>
#include "defines.hpp"
#include "matrix.hpp"

namespace capstone {
namespace base {

class Dataset {
public:
    explicit Dataset(const std::string& filename,
                     const DATATYPE& dataType,
                     const uint32_t& nImages = NIMAGES);
    virtual ~Dataset() = default;
    virtual void init() = 0;
    std::string show();
    void wait();
    inline const std::string& getFilename() const {
        return m_filename;
    }
    inline const uint32_t& getMagicNumber() const {
        return m_magicNumber;
    }
    inline const uint32_t& getNImages() const {
        return m_nImages;
    }
    inline const DATATYPE& getDataType() const {
        return m_dataType;
    }
    inline const std::string& getDataTypeStr() const {
        return m_dataTypeStr;
    }

protected:
    Synch_t m_synch;
    std::string m_filename;
    uint32_t m_magicNumber;
    uint32_t m_nImages;
    uint32_t m_nImagesTotal;
    DATATYPE m_dataType;
    std::string m_dataTypeStr;
    double m_mu;
    double m_sigma2;
};

class DatasetImage final : public Dataset {
public:
    explicit DatasetImage(const std::string& filename,
                          const DATATYPE& dataType);
    ~DatasetImage() final = default;
    void init() final;
    std::string showIndex(const int& index);
    Matrix& operator()(const int& i) {
        return m_data[i];
    }
    const Matrix& operator()(const int& i) const {
        return m_data[i];
    }

private:
    std::vector<Matrix> m_data;
    void pad(const uint32_t& nrows,
             const uint32_t& p,
             std::vector<double>& data);
};

class DatasetLabel final : public Dataset {
public:
    explicit DatasetLabel(const std::string& filename,
                          const DATATYPE& dataType);
    ~DatasetLabel() final = default;
    void init() final;
    std::string showIndex(const int& index);
    inline unsigned char& operator()(const int& i) {
        return m_data[i];
    }
    const unsigned char& operator()(const int& i) const {
        return m_data[i];
    }

private:
    std::vector<unsigned char> m_data;
};

typedef std::pair<unsigned char, Matrix> Datapoint_t;

class DatapointSet {
public:
    DatapointSet(const DatasetImage& images,
                 const DatasetLabel& labels);
    ~DatapointSet() = default;
    void shuffle();
    inline Datapoint_t& nextTrainData() {
        assert(!m_shuffledIndices.empty());
        int index = m_shuffledIndices.back();
        m_shuffledIndices.pop_back();
        assert(index < m_trainSet.size());
        return m_trainSet[index];
    }
    inline Datapoint_t& validData(const int& index) {
        assert(index < m_validSet.size());
        return m_validSet[index];
    }
    inline const std::vector<Datapoint_t>& getTrainData() const {
        return m_trainSet;
    }
    inline const std::vector<Datapoint_t>& getValidData() const {
        return m_validSet;
    }
    inline const uint32_t getTrainSize() const {
        return m_trainSet.size();
    }
    inline const uint32_t getValidSize() const {
        return m_validSet.size();
    }

private:
    std::vector<int> m_shuffledIndices;
    std::vector<Datapoint_t> m_trainSet;
    std::vector<Datapoint_t> m_validSet;

};

} /* base */
} /* capstone */
#endif /* INC_DATASET_HPP_ */
