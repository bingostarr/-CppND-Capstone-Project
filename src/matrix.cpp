/*
 * matrix.cpp
 *
 *  Created on: Feb 13, 2020
 *      Author: bingo
 */

#include "matrix.hpp"
#include <cassert>
#include <algorithm>

namespace capstone {
namespace base {

Matrix::Matrix(const ImageSize_t& size,
               const MTXTYPE& mtype)
        : m_size(size),
          m_matrix {std::make_unique<double[]>(m_size.nPixels())} {
    for (int i = 0; i < getSize(); ++i) {
        for (int j = 0; j < getSize(); ++j) {
            switch (mtype) {
            case MTXTYPE::ONES:
                at(i,j) = 1.0;
                break;
            case MTXTYPE::ID:
                at(i,j) = (i == j) ? 1.0 : 0.0;
                break;
            default:
                at(i,j) = 0.0;
            }
        }
    }
}

Matrix::Matrix(const ImageSize_t& size,
               const std::vector<double>& v)
        : m_size(size),
          m_matrix {std::make_unique<double[]>(m_size.nPixels())} {
    assert(v.size() == m_size.nPixels());
    int k = 0;
    for (int i = 0; i < getSize(); ++i) {
        for (int j = 0; j < getSize(); ++j) {
            at(i,j) = v[k];
            k++;
        }
    }
}

Matrix::Matrix(const Matrix& m)
        : m_size(m.getSize()),
          m_matrix {std::make_unique<double[]>(m_size.nPixels())} {
    for (int i = 0; i < getSize(); ++i) {
        for (int j = 0; j < getSize(); ++j) {
            at(i,j) = m(i, j);
        }
    }
}

Matrix::Matrix(Matrix&& m)
        : m_size(m.getSize()),
          m_matrix {std::make_unique<double[]>(m_size.nPixels())} {
    int k = 0;
    for (int i = 0; i < getSize(); ++i) {
        for (int j = 0; j < getSize(); ++j) {
            m_matrix[k] = m(i, j);
            k++;
        }
    }
}

const Matrix Matrix::operator=(const Matrix& m) const {
    Matrix x(m);
//    Matrix x(m_size, MTXTYPE::ZEROS);
//    for (int i = 0; i < getSize(); ++i) {
//        for (int j = 0; j < getSize(); ++j) {
//            x(i,j) = m(i,j);
//        }
//    }
    return x;
}

const Matrix Matrix::operator=(Matrix&& m) const {
    Matrix x(m);
//    Matrix x(m_size, MTXTYPE::ZEROS);
//    for (int i = 0; i < getSize(); ++i) {
//        for (int j = 0; j < getSize(); ++j) {
//            x(i,j) = m(i,j);
//        }
//    }
    return x;
}

const Matrix Matrix::operator+(const Matrix& m) const {
    assert(m.getSize() == getSize());
    Matrix x(m_size, MTXTYPE::ZEROS);
    for (int i = 0; i < getSize(); ++i) {
        for (int j = 0; j < getSize(); ++j) {
            x(i,j) = at(i,j) + m(i,j);
        }
    }
    return x;
}

const Matrix Matrix::operator+(Matrix&& m) const {
    assert(m.getSize() == getSize());
    Matrix x(m_size, MTXTYPE::ZEROS);
    for (int i = 0; i < getSize(); ++i) {
        for (int j = 0; j < getSize(); ++j) {
            x(i,j) = at(i,j) + m(i,j);
        }
    }
    return x;
}

const Matrix Matrix::operator-(const Matrix& m) const {
    assert(m.getSize() == getSize());
    Matrix x(m_size, MTXTYPE::ZEROS);
    for (int i = 0; i < getSize(); ++i) {
        for (int j = 0; j < getSize(); ++j) {
            x(i,j) = at(i,j) - m(i,j);
        }
    }
    return x;
}

const Matrix Matrix::operator-(Matrix&& m) const {
    assert(m.getSize() == getSize());
    Matrix x(m_size, MTXTYPE::ZEROS);
    for (int i = 0; i < getSize(); ++i) {
        for (int j = 0; j < getSize(); ++j) {
            x(i,j) = at(i,j) - m(i,j);
        }
    }
    return x;
}

const Matrix Matrix::operator*(const Matrix& m) const {
    assert(m.getSize() == getSize());
    Matrix x(m_size, MTXTYPE::ZEROS);
    for (int i = 0; i < getSize(); ++i) {
        for (int j = 0; j < getSize(); ++j) {
            x(i,j) = at(i,j) * m(i,j);
        }
    }
    return x;
}

const Matrix Matrix::operator*(Matrix&& m) const {
    assert(m.getSize() == getSize());
    Matrix x(m_size, MTXTYPE::ZEROS);
    for (int i = 0; i < getSize(); ++i) {
        for (int j = 0; j < getSize(); ++j) {
            x(i,j) = at(i,j) * m(i,j);
        }
    }
    return x;
}

const Matrix Matrix::operator/(const Matrix& m) const {
    assert(m.getSize() == getSize());
    Matrix x(m_size, MTXTYPE::ZEROS);
    for (int i = 0; i < getSize(); ++i) {
        for (int j = 0; j < getSize(); ++j) {
            x(i,j) = at(i,j) / m(i,j);
        }
    }
    return x;
}

const Matrix Matrix::operator/(Matrix&& m) const {
    assert(m.getSize() == getSize());
    Matrix x(m_size, MTXTYPE::ZEROS);
    for (int i = 0; i < getSize(); ++i) {
        for (int j = 0; j < getSize(); ++j) {
            x(i,j) = at(i,j) / m(i,j);
        }
    }
    return x;
}

const Matrix Matrix::operator+(const double& d) const {
    Matrix x(m_size, MTXTYPE::ZEROS);
    for (int i = 0; i < getSize(); ++i) {
        for (int j = 0; j < getSize(); ++j) {
            x(i,j) = at(i,j) + d;
        }
    }
    return x;
}

const Matrix Matrix::operator-(const double& d) const {
    Matrix x(m_size, MTXTYPE::ZEROS);
    for (int i = 0; i < getSize(); ++i) {
        for (int j = 0; j < getSize(); ++j) {
            x(i,j) = at(i,j) - d;
        }
    }
    return x;
}

const Matrix Matrix::operator*(const double& d) const {
    Matrix x(m_size, MTXTYPE::ZEROS);
    for (int i = 0; i < getSize(); ++i) {
        for (int j = 0; j < getSize(); ++j) {
            x(i,j) = at(i,j) * d;
        }
    }
    return x;
}

const Matrix Matrix::operator/(const double& d) const {
    Matrix x(m_size, MTXTYPE::ZEROS);
    for (int i = 0; i < getSize(); ++i) {
        for (int j = 0; j < getSize(); ++j) {
            x(i,j) = at(i,j) / d;
        }
    }
    return x;
}

double& Matrix::operator()(const int& i,
                           const int& j) {
    assert(m_size.inRange(i, j));
    return m_matrix[(i * getSize()) + j];
}

const double& Matrix::operator()(const int& i,
                                 const int& j) const {
    assert(m_size.inRange(i, j));
    return m_matrix[(i * getSize()) + j];
}

double& Matrix::at(const int& i,
                   const int& j) {
    assert(m_size.inRange(i, j));
    return m_matrix[(i * getSize()) + j];
}

const double& Matrix::at(const int& i,
                         const int& j) const {
    assert(m_size.inRange(i, j));
    return m_matrix[(i * getSize()) + j];
}

std::string Matrix::show() {
    std::string x = "{\n";
    int k = 0;
    for (int i = 0; i < getSize(); ++i) {
        x += "\t";
        for (int j = 0; j < getSize(); ++j) {
            std::string s = std::to_string(m_matrix[k]);
            x += s + "\t";
            k++;
        }
        x += "\n";
    }
    x += "}--"+ std::to_string(getSize()) + "X" + std::to_string(getSize());
    return x;
}

const Matrix Matrix::subMatrix(const int& a,
                               const int& b,
                               const ImageSize_t& size) const
{
    assert(m_size.inRange(a, b));
    assert(size < m_size);
    assert((a + m_size.getSize()) <= getSize());
    assert((b + m_size.getSize()) <= getSize());
    Matrix x(size, MTXTYPE::ZEROS);
    for (int i = 0; i < size.getSize(); ++i) {
        for (int j = 0; j < size.getSize(); ++j) {
            x(i,j) = at(a + i, b + j);
        }
    }
    return x;
}

const double Matrix::sum() const {
    double c = 0;
    for (int i = 0; i < getSize() * getSize(); ++i) {
        c += m_matrix[i];
    }
    return c;
}

const Coords_t Matrix::getIndexMax() const {
    std::vector<double> v = vectorize();
    int i = std::max_element(v.begin(), v.end()) - v.begin();
    Coords_t c(i / getSize(), i % getSize());
    return c;
}

const double& Matrix::max() const {
    Coords_t c = getIndexMax();
    return at(c.i, c.j);
}

const Coords_t Matrix::getIndexMin() const {
    std::vector<double> v = vectorize();
    int i = std::min_element(v.begin(), v.end()) - v.begin();
    Coords_t c(i / getSize(), i % getSize());
    return c;
}

const double& Matrix::min() const {
    Coords_t c = getIndexMin();
    return at(c.i, c.j);
}

const Matrix Matrix::transpose() const {
    Matrix x(m_size, MTXTYPE::ZEROS);
    int k = 0;
    for (int i = 0; i < getSize(); ++i) {
        for (int j = 0; j < getSize(); ++j) {
            x(j,i) = m_matrix[k];
            k++;
        }
    }
    return x;
}

const Matrix Matrix::product(Matrix& m) const {
    assert(m.getSize() == getSize());
    Matrix x(getSize(), MTXTYPE::ZEROS);
    for (int i = 0; i < x.getSize(); ++i) {
        for (int j = 0; j < x.getSize(); ++j) {
            double c = 0;
            for (int k = 0; k < getSize(); ++k) {
                c += at(i,k) * m(k, j);
            }
            x(i,j) = c;
        }
    }
    return x;
}

const std::vector<double> Matrix::vectorize() const
{
    std::vector<double> v{};
    for (int i = 0; i < getSize(); ++i) {
        for (int j = 0; j < getSize(); ++j) {
            v.push_back(at(i,j));
        }
    }
    return v;
}

} /* base */
} /* capstone */
