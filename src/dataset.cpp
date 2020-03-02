/*
 * dataset.cpp
 *
 *  Created on: Feb 13, 2020
 *      Author: bingo
 */

#include "dataset.hpp"
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <thread>
#include <cmath>
#include <random>
#include <algorithm>

namespace capstone {
namespace base {

static std::random_device g_device;
static std::mt19937 g_generator(g_device());

Dataset::Dataset(const std::string& filename,
                 const DATATYPE& dataType,
                 const uint32_t& nImages)
        : m_synch{},
          m_filename(filename),
          m_magicNumber(0),
          m_nImages(nImages),
          m_nImagesTotal(0),
          m_dataType(dataType),
          m_dataTypeStr((dataType == DATATYPE::TEST)? "test": "train"),
          m_mu(0),
          m_sigma2(0) {
    assert(nImages <= NIMAGESMAX);
}

void Dataset::wait() {
    m_synch.wait();
}

std::string Dataset::show() {
    std::string x = "{\n";
    x += "\tfilename: " + m_filename+ "\n";
    x += "\ttype: " + m_dataTypeStr + "\n";
    x += "\tmagic number: " + std::to_string(m_magicNumber)+ "\n";
    x += "\timages: " + std::to_string(m_nImages)+ "\n";
    x += "\timagesTotal: " + std::to_string(m_nImagesTotal)+ "\n";
    x += "\tmean: " + std::to_string(m_mu)+ "\n";
    x += "\tstd. dev.: " + std::to_string(m_sigma2)+ "\n";
    x += "}";
    return x;
}

DatasetImage::DatasetImage(const std::string& filename,
                           const DATATYPE& dataType)
        : Dataset(filename, dataType) {
    std::thread a(&DatasetImage::init, this);
    a.join();
}

void DatasetImage::init() {
    assert(access(m_filename.c_str(), F_OK) != -1);
    std::ifstream input(m_filename.c_str(),
                        std::ios::in | std::ios::binary);
    assert(!input.fail());
    m_data.clear();
    m_magicNumber = 0;
    m_nImagesTotal = 0;
    uint32_t nrows = 0;
    uint32_t ncols = 0;
    std::vector<double> data{};
    char c;
    int count = 0;
    int pixelCount = 0;
    int nImageCount = 0;
    std::vector<double> rawData{};
    while (input.get(c)) {
        uint8_t k = static_cast<uint8_t>(c);
        if ((count >= 0) && (count < 4)) {
            m_magicNumber = (m_magicNumber << 8) | k;
        }
        else if ((count >= 4) && (count < 8)) {
            m_nImagesTotal = (m_nImagesTotal << 8) | k;
        }
        else if ((count >= 8) && (count < 12)) {
            nrows = (nrows << 8) | k;
        }
        else if ((count >= 12) && (count < 16)) {
            ncols = (ncols << 8) | k;
        }
        else {
            assert(nrows == ncols); // square matrices only.
            assert((nrows % 2) == 0); // even number of rows and cols
            data.push_back(static_cast<double>(k) / SCALE);
            rawData.push_back(static_cast<double>(k) / SCALE);
            pixelCount++;
            if (pixelCount == (nrows * ncols)) {
                ImageSize_t imgSize(INPUTSIZE);
                if (nrows < INPUTSIZE) {
                    uint32_t p = (INPUTSIZE - nrows) / 2;
                    pad(nrows, p, data);
                }
                Matrix m(imgSize, data);
                m_data.push_back(m);
                nImageCount++;
                data.clear();
                pixelCount = 0;
                if (nImageCount == m_nImages) {
                    break;
                }
            }
        }
        count++;
    }
    input.close();
    int n = rawData.size();
    m_mu = 0;
    m_sigma2 = 0;
    for (double d : rawData) {
        m_mu += d;
    }
    m_mu /= n;
    for (double d : rawData) {
        m_sigma2 += ((d - m_mu) * (d - m_mu)) / n;
    }
    m_sigma2 = sqrt(m_sigma2);
    for (int i = 0; i < m_data.size(); ++i) {
        m_data[i].normalize(m_mu, m_sigma2);
    }
    assert(m_nImages == m_data.size());
    m_synch.set();
}

std::string DatasetImage::showIndex(const int& index) {
    assert(index < m_nImages);
    return (m_data[index].show());
}

void DatasetImage::pad(const uint32_t& nrows,
                       const uint32_t& p,
                       std::vector<double>& data) {
    assert(p >= 0);
    if (p == 0) {
        return;
    }
    int count1 = 0;
    std::vector<double> tdata;
    for (int i = 0; i < (nrows + 2 * p); ++i) {
        for (int j = 0; j < (nrows + 2 * p); ++j) {
            if ((i >= p) && (j >= p) && (i < (nrows + p)) && (j < (nrows + p))) {
                tdata.push_back(data[count1]);
                count1++;
            }
            else {
                tdata.push_back(0);
            }
        }
    }
    data = std::move(tdata);
}

DatasetLabel::DatasetLabel(const std::string& filename,
                           const DATATYPE& dataType)
        : Dataset(filename, dataType) {
    std::thread a(&DatasetLabel::init, this);
    a.join();
}

void DatasetLabel::init() {
    assert(access(m_filename.c_str(), F_OK) != -1);
    std::ifstream input(m_filename.c_str(),
                        std::ios::in | std::ios::binary);
    assert(!input.fail());
    m_data.clear();
    m_magicNumber = 0;
    m_nImagesTotal = 0;
    char c;
    int count = 0;
    int nImageCount = 0;
    while (input.get(c)) {
        uint8_t k = static_cast<uint8_t>(c);
        if ((count >= 0) && (count < 4)) {
            m_magicNumber = (m_magicNumber << 8) | k;
        }
        else if ((count >= 4) && (count < 8)) {
            m_nImagesTotal = (m_nImagesTotal << 8) | k;
        }
        else {
            m_data.push_back(k);
            nImageCount++;
            if (nImageCount == m_nImages) {
                break;
            }
        }
        count++;
    }
    input.close();
    assert(m_nImages == m_data.size());
    m_synch.set();
}

std::string DatasetLabel::showIndex(const int& index) {
    assert(index < m_nImages);
    return std::to_string(m_data[index]);
}

DatapointSet::DatapointSet(const DatasetImage& images,
                           const DatasetLabel& labels)
        : m_shuffledIndices({ }),
          m_trainSet({ }),
          m_validSet({ }) {
    assert(images.getNImages() == labels.getNImages());
    assert(images.getDataType() == labels.getDataType());
    DATATYPE dType = labels.getDataType();
    uint32_t nImages = labels.getNImages();
    uint32_t nTrain = 0;
    uint32_t nValid = nImages;
    if (DATATYPE::TRAIN == dType) {
        assert((SPLIT >= 0) && (SPLIT <= 1.0));
        nTrain = nImages * SPLIT;
        nValid -= nTrain;
    }
    int count = 0;
    for (int i = 0; i < nTrain; ++i) {
        m_trainSet.push_back(std::make_pair(labels(count), images(count)));
        count++;
    }
    for (int i = 0; i < nValid; ++i) {
        m_validSet.push_back(std::make_pair(labels(count), images(count)));
        count++;
    }
    shuffle();
}

void DatapointSet::shuffle() {
    if (m_trainSet.empty()) {
        return;
    }
    m_shuffledIndices.clear();
    for (int i = 0; i < m_trainSet.size(); ++i) {
        m_shuffledIndices.push_back(i);
    }
    std::shuffle(m_shuffledIndices.begin(), m_shuffledIndices.end(), g_generator);
}

} /* base */
} /* capstone */
