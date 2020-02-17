/*
 * dataset.cpp
 *
 *  Created on: Feb 13, 2020
 *      Author: bingo
 */

#include "dataset.hpp"
#include <unistd.h>
#include <cassert>
#include <fstream>
#include <thread>

namespace capstone {
namespace base {

Dataset::Dataset(const std::string& filename)
        : m_synch{},
          m_filename(filename),
          m_magicNumber(0),
          m_nImages(0) {
}

void Dataset::wait() {
    m_synch.wait();
}

std::string Dataset::show() {
    std::string x = "{\n";
    x += "\tmagic number: " + std::to_string(m_magicNumber)+ "\n";
    x += "\timages: " + std::to_string(m_nImages)+ "\n";
    x += "}";
    return x;
}

DatasetImage::DatasetImage(const std::string& filename)
        : Dataset(filename) {
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
    m_nImages = 0;
    uint32_t nrows = 0;
    uint32_t ncols = 0;
    std::vector<double> data{};
    char c;
    int count = 0;
    int pixelCount = 0;
    while (input.get(c)) {
        uint8_t k = static_cast<uint8_t>(c);
        if ((count >= 0) && (count < 4)) {
            m_magicNumber = (m_magicNumber << 8) | k;
        }
        else if ((count >= 4) && (count < 8)) {
            m_nImages = (m_nImages << 8) | k;
        }
        else if ((count >= 8) && (count < 12)) {
            nrows = (nrows << 8) | k;
        }
        else if ((count >= 12) && (count < 16)) {
            ncols = (ncols << 8) | k;
        }
        else {
            ImageSize_t imgSize(nrows, ncols);
            data.push_back(static_cast<double>(k));
            pixelCount++;
            if (pixelCount == (nrows * ncols)) {
                Matrix m(imgSize, data);
                m_data.push_back(m);
                data.clear();
                pixelCount = 0;
            }
        }
        count++;
    }
    input.close();
    assert(m_nImages == m_data.size());
    m_synch.set();
}

std::string DatasetImage::showIndex(const int& index) {
    assert(index < m_nImages);
    return (m_data[index].show());
}

DatasetLabel::DatasetLabel(const std::string& filename)
        : Dataset(filename) {
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
    m_nImages = 0;
    std::vector<double> data{};
    char c;
    int count = 0;
    while (input.get(c)) {
        uint8_t k = static_cast<uint8_t>(c);
        if ((count >= 0) && (count < 4)) {
            m_magicNumber = (m_magicNumber << 8) | k;
        }
        else if ((count >= 4) && (count < 8)) {
            m_nImages = (m_nImages << 8) | k;
        }
        else {
            m_data.push_back(k);
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

} /* base */
} /* capstone */
