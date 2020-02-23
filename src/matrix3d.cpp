/*
 * matrix3d.cpp
 *
 *  Created on: Feb 22, 2020
 *      Author: bingo
 */

#include "matrix3d.hpp"
#include <cassert>

namespace capstone {
namespace base {

Matrix3d::Matrix3d(const CubeSize_t& size)
        : m_size(size) {
    m_matrix3d.clear();
    for (int i = 0; i < getNLayers(); ++i) {
        Matrix x(getImageSize());
        m_matrix3d.push_back(x);
    }
}

Matrix3d::Matrix3d(const Matrix3d& m)
        : m_size(m.getSize()) {
    m_matrix3d.clear();
    for (int i = 0; i < getNLayers(); ++i) {
        Matrix x(m.at(i));
        m_matrix3d.push_back(x);
    }
}

Matrix3d::Matrix3d(Matrix3d&& m)
        : m_size(m.getSize()) {
    m_matrix3d.clear();
    for (int i = 0; i < getNLayers(); ++i) {
        Matrix x(m.at(i));
        m_matrix3d.push_back(x);
    }
}

const Matrix3d Matrix3d::operator=(const Matrix3d& m) const {
    Matrix3d x(m);
    return x;
}

const Matrix3d Matrix3d::operator=(Matrix3d&& m) const {
    Matrix3d x(m);
    return x;
}

const Matrix3d Matrix3d::operator+(const Matrix3d& m) const {
    assert(m.getSize() == getSize());
    Matrix3d x(m_size);
    for (int i = 0; i < getNLayers(); ++i) {
        x.at(i) = at(i) + m.at(i);
    }
    return x;
}

const Matrix3d Matrix3d::operator+(Matrix3d&& m) const {
    assert(m.getSize() == getSize());
    Matrix3d x(m_size);
    for (int i = 0; i < getNLayers(); ++i) {
            x.at(i) = at(i) + m.at(i);
    }
    return x;
}

const Matrix3d Matrix3d::operator-(const Matrix3d& m) const {
    assert(m.getSize() == getSize());
    Matrix3d x(m_size);
    for (int i = 0; i < getNLayers(); ++i) {
            x.at(i) = at(i) - m.at(i);
    }
    return x;
}

const Matrix3d Matrix3d::operator-(Matrix3d&& m) const {
    assert(m.getSize() == getSize());
    Matrix3d x(m_size);
    for (int i = 0; i < getNLayers(); ++i) {
            x.at(i) = at(i) - m.at(i);
    }
    return x;
}

const Matrix3d Matrix3d::operator*(const Matrix3d& m) const {
    assert(m.getSize() == getSize());
    Matrix3d x(m_size);
    for (int i = 0; i < getNLayers(); ++i) {
            x.at(i) = at(i) * m.at(i);
    }
    return x;
}

const Matrix3d Matrix3d::operator*(Matrix3d&& m) const {
    assert(m.getSize() == getSize());
    Matrix3d x(m_size);
    for (int i = 0; i < getNLayers(); ++i) {
            x.at(i) = at(i) * m.at(i);
    }
    return x;
}

const Matrix3d Matrix3d::operator/(const Matrix3d& m) const {
    assert(m.getSize() == getSize());
    Matrix3d x(m_size);
    for (int i = 0; i < getNLayers(); ++i) {
            x.at(i) = at(i) / m.at(i);
    }
    return x;
}

const Matrix3d Matrix3d::operator/(Matrix3d&& m) const {
    assert(m.getSize() == getSize());
    Matrix3d x(m_size);
    for (int i = 0; i < getNLayers(); ++i) {
            x.at(i) = at(i) / m.at(i);
    }
    return x;
}

const Matrix3d Matrix3d::operator+(const double& d) const {
    Matrix3d x(m_size);
    for (int i = 0; i < getNLayers(); ++i) {
            x.at(i) = at(i) + d;
    }
    return x;
}

const Matrix3d Matrix3d::operator-(const double& d) const {
    Matrix3d x(m_size);
    for (int i = 0; i < getNLayers(); ++i) {
            x.at(i) = at(i) - d;
    }
    return x;
}

const Matrix3d Matrix3d::operator*(const double& d) const {
    Matrix3d x(m_size);
    for (int i = 0; i < getNLayers(); ++i) {
            x.at(i) = at(i) * d;
    }
    return x;
}

const Matrix3d Matrix3d::operator/(const double& d) const {
    Matrix3d x(m_size);
    for (int i = 0; i < getNLayers(); ++i) {
            x.at(i) = at(i) / d;
    }
    return x;
}

double& Matrix3d::operator()(const int& k,
                             const int& i,
                             const int& j) {
    assert(m_size.inRange(k, i, j));
    return at(k).at(i,j);
}

const double& Matrix3d::operator()(const int& k,
                                   const int& i,
                                   const int& j) const {
    assert(m_size.inRange(k, i, j));
    return at(k).at(i,j);
}

Matrix& Matrix3d::at(const int& k) {
    assert(k < getNLayers());
    return m_matrix3d[k];
}

const Matrix& Matrix3d::at(const int& k) const {
    assert(k < getNLayers());
    return m_matrix3d[k];
}

std::string Matrix3d::show() {
    std::string x = "{\n";
    for (int i = 0; i < getNLayers(); ++i) {
        x += at(i).show() + "\n";
    }
    x += "}--"+ m_size.show();
    return x;
}

const std::vector<double> Matrix3d::vectorize() const
{
    assert(getSize().getImageSize().getSize() == 1);
    std::vector<double> v{};
    const int i = 0;
    for (int k = 0; k < getNLayers(); ++k) {
        v.push_back(m_matrix3d[k].at(i,i));
    }
    return v;
}

} /* base */
} /* capstone */
