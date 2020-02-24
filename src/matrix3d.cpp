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

Matrix3d::Matrix3d(const CubeSize_t& size,
                   const MTXTYPE& mtxType)
        : m_size(size) {
    m_matrix3d.clear();
    for (int i = 0; i < getNLayers(); ++i) {
        Matrix x(getImageSize(), mtxType);
        m_matrix3d.push_back(x);
    }
}

Matrix3d::Matrix3d(const Matrix3d& m)
        : m_size(m.getCubeSize()) {
    m_matrix3d.clear();
    for (int i = 0; i < getNLayers(); ++i) {
        Matrix x(m.at(i));
        m_matrix3d.push_back(x);
    }
}

Matrix3d::Matrix3d(Matrix3d&& m)
        : m_size(m.getCubeSize()) {
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
    assert(m.getCubeSize() == getCubeSize());
    Matrix3d x(m_size);
    for (int i = 0; i < getNLayers(); ++i) {
        x.at(i) = at(i) + m.at(i);
    }
    return x;
}

const Matrix3d Matrix3d::operator+(Matrix3d&& m) const {
    assert(m.getCubeSize() == getCubeSize());
    Matrix3d x(m_size);
    for (int i = 0; i < getNLayers(); ++i) {
            x.at(i) = at(i) + m.at(i);
    }
    return x;
}

const Matrix3d Matrix3d::operator-(const Matrix3d& m) const {
    assert(m.getCubeSize() == getCubeSize());
    Matrix3d x(m_size);
    for (int i = 0; i < getNLayers(); ++i) {
            x.at(i) = at(i) - m.at(i);
    }
    return x;
}

const Matrix3d Matrix3d::operator-(Matrix3d&& m) const {
    assert(m.getCubeSize() == getCubeSize());
    Matrix3d x(m_size);
    for (int i = 0; i < getNLayers(); ++i) {
            x.at(i) = at(i) - m.at(i);
    }
    return x;
}

const Matrix3d Matrix3d::operator*(const Matrix3d& m) const {
    assert(m.getCubeSize() == getCubeSize());
    Matrix3d x(m_size);
    for (int i = 0; i < getNLayers(); ++i) {
            x.at(i) = at(i) * m.at(i);
    }
    return x;
}

const Matrix3d Matrix3d::operator*(Matrix3d&& m) const {
    assert(m.getCubeSize() == getCubeSize());
    Matrix3d x(m_size);
    for (int i = 0; i < getNLayers(); ++i) {
            x.at(i) = at(i) * m.at(i);
    }
    return x;
}

const Matrix3d Matrix3d::operator/(const Matrix3d& m) const {
    assert(m.getCubeSize() == getCubeSize());
    Matrix3d x(m_size);
    for (int i = 0; i < getNLayers(); ++i) {
            x.at(i) = at(i) / m.at(i);
    }
    return x;
}

const Matrix3d Matrix3d::operator/(Matrix3d&& m) const {
    assert(m.getCubeSize() == getCubeSize());
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

double& Matrix3d::at(const int& k,
                     const int& i,
                     const int& j) {
    assert(m_size.inRange(k, i, j));
    return at(k).at(i,j);
}

const double& Matrix3d::at(const int& k,
                           const int& i,
                           const int& j) const {
    assert(m_size.inRange(k, i, j));
    return at(k).at(i,j);
}

std::string Matrix3d::show() {
    std::string x = "{\n";
    for (int i = 0; i < getNLayers(); ++i) {
        x += at(i).show() + "\n";
    }
    x += "}--"+ m_size.show();
    return x;
}

const double Matrix3d::sum() const {
    double c = 0;
    for (int k = 0; k < m_size.getNLayers(); ++k) {
        c += at(k).sum();
    }
    return c;
}

const Matrix3d Matrix3d::subMatrix3d(const int& c,
                                     const int& a,
                                     const int& b,
                                     const CubeSize_t& size) const {
    assert(m_size.inRange(c, a, b));
    assert(size < m_size);
    assert((c + size.getNLayers()) <= getNLayers());
    assert((a + size.getSize()) <= getSize());
    assert((b + size.getSize()) <= getSize());
    Matrix3d x(size);
    for (int k = 0; k < size.getNLayers(); ++k) {
        for (int i = 0; i < size.getSize(); ++i) {
            for (int j = 0; j < size.getSize(); ++j) {
                x(k, i, j) = at(c + k, a + i, b + j);
            }
        }
    }
    return x;
}

void Matrix3d::fillSubMatrix3d(const int& c,
                               const int& a,
                               const int& b,
                               const Matrix3d& m) {
    assert(m_size.inRange(c, a, b));
    assert(m.getCubeSize() < m_size);
    assert((c + m.getNLayers()) <= getNLayers());
    assert((a + m.getSize()) <= getSize());
    assert((b + m.getSize()) <= getSize());
    for (int k = c; k < m.getNLayers(); ++k) {
        for (int i = a; i < m.getSize(); ++i) {
            for (int j = b; j < m.getSize(); ++j) {
                at(k, i, j) = m.at(k - c, i - a, j - b);
            }
        }
    }
}

const std::vector<double> Matrix3d::vectorize() const
{
    assert(getSize() == 1);
    std::vector<double> v{};
    for (int k = 0; k < getNLayers(); ++k) {
        v.push_back(m_matrix3d[k].at(0,0));
    }
    return v;
}

void Matrix3d::zero() {
    for (int k = 0; k < getNLayers(); ++k) {
        m_matrix3d[k].zero();
    }
}

} /* base */
} /* capstone */
