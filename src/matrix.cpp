/*
 * matrix.cpp
 *
 *  Created on: Feb 13, 2020
 *      Author: bingo
 */

#include "matrix.hpp"
#include <cassert>

namespace capstone {
namespace base {

Matrix::Matrix(const ImageSize_t& size,
               const MTX_TYPE& mtype)
        : m_size(size),
          m_matrix {std::make_unique<double[]>(m_size.npixels)} {
    for (int i = 0; i < getRows(); ++i) {
        for (int j = 0; j < getCols(); ++j) {
            switch (mtype) {
            case MTX_TYPE::ONES:
                at(i,j) = 1.0;
                break;
            case MTX_TYPE::ID:
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
          m_matrix {std::make_unique<double[]>(m_size.npixels)} {
    assert(v.size() == m_size.npixels);
    int k = 0;
    for (int i = 0; i < getRows(); ++i) {
        for (int j = 0; j < getCols(); ++j) {
            at(i,j) = v[k];
            k++;
        }
    }
}

Matrix::Matrix(const Matrix& m)
        : m_size(m.getSize()),
          m_matrix {std::make_unique<double[]>(m_size.npixels)} {
    for (int i = 0; i < getRows(); ++i) {
        for (int j = 0; j < getCols(); ++j) {
            at(i,j) = m(i, j);
        }
    }
}

Matrix::Matrix(Matrix&& m)
        : m_size(m.getSize()),
          m_matrix {std::make_unique<double[]>(m_size.npixels)} {
    int k = 0;
    for (int i = 0; i < getRows(); ++i) {
        for (int j = 0; j < getCols(); ++j) {
            m_matrix[k] = m(i, j);
            k++;
        }
    }
}

const Matrix Matrix::operator=(const Matrix& m) const {
    assert(m.getSize() == getSize());
    Matrix x(m_size, MTX_TYPE::ZEROS);
    for (int i = 0; i < getRows(); ++i) {
        for (int j = 0; j < getCols(); ++j) {
            x(i,j) = m(i,j);
        }
    }
    return x;
}

const Matrix Matrix::operator=(Matrix&& m) const {
    assert(m.getSize() == getSize());
    Matrix x(m_size, MTX_TYPE::ZEROS);
    for (int i = 0; i < getRows(); ++i) {
        for (int j = 0; j < getCols(); ++j) {
            x(i,j) = m(i,j);
        }
    }
    return x;
}

const Matrix Matrix::operator+(const Matrix& m) const {
    assert(m.getSize() == getSize());
    Matrix x(m_size, MTX_TYPE::ZEROS);
    for (int i = 0; i < getRows(); ++i) {
        for (int j = 0; j < getCols(); ++j) {
            x(i,j) = at(i,j) + m(i,j);
        }
    }
    return x;
}

const Matrix Matrix::operator+(Matrix&& m) const {
    assert(m.getSize() == getSize());
    Matrix x(m_size, MTX_TYPE::ZEROS);
    for (int i = 0; i < getRows(); ++i) {
        for (int j = 0; j < getCols(); ++j) {
            x(i,j) = at(i,j) + m(i,j);
        }
    }
    return x;
}

const Matrix Matrix::operator-(const Matrix& m) const {
    assert(m.getSize() == getSize());
    Matrix x(m_size, MTX_TYPE::ZEROS);
    for (int i = 0; i < getRows(); ++i) {
        for (int j = 0; j < getCols(); ++j) {
            x(i,j) = at(i,j) - m(i,j);
        }
    }
    return x;
}

const Matrix Matrix::operator-(Matrix&& m) const {
    assert(m.getSize() == getSize());
    Matrix x(m_size, MTX_TYPE::ZEROS);
    for (int i = 0; i < getRows(); ++i) {
        for (int j = 0; j < getCols(); ++j) {
            x(i,j) = at(i,j) - m(i,j);
        }
    }
    return x;
}

const Matrix Matrix::operator*(const Matrix& m) const {
    assert(m.getSize() == getSize());
    Matrix x(m_size, MTX_TYPE::ZEROS);
    int k = 0;
    for (int i = 0; i < getRows(); ++i) {
        for (int j = 0; j < getCols(); ++j) {
            x(i,j) = at(i,j) * m(i,j);
            k++;
        }
    }
    return x;
}

const Matrix Matrix::operator*(Matrix&& m) const {
    assert(m.getSize() == getSize());
    Matrix x(m_size, MTX_TYPE::ZEROS);
    int k = 0;
    for (int i = 0; i < getRows(); ++i) {
        for (int j = 0; j < getCols(); ++j) {
            x(i,j) = at(i,j) * m(i,j);
            k++;
        }
    }
    return x;
}

const Matrix Matrix::operator*(const double& d) const {
    Matrix x(m_size, MTX_TYPE::ZEROS);
    int k = 0;
    for (int i = 0; i < getRows(); ++i) {
        for (int j = 0; j < getCols(); ++j) {
            x(i,j) = d * at(i,j);
            k++;
        }
    }
    return x;
}

const Matrix Matrix::operator/(const Matrix& m) const {
    assert(m.getSize() == getSize());
    Matrix x(m_size, MTX_TYPE::ZEROS);
    int k = 0;
    for (int i = 0; i < getRows(); ++i) {
        for (int j = 0; j < getCols(); ++j) {
            x(i,j) = at(i,j) / m(i,j);
            k++;
        }
    }
    return x;
}

const Matrix Matrix::operator/(Matrix&& m) const {
    assert(m.getSize() == getSize());
    Matrix x(m_size, MTX_TYPE::ZEROS);
    for (int i = 0; i < getRows(); ++i) {
        for (int j = 0; j < getCols(); ++j) {
            x(i,j) = at(i,j) / m(i,j);
        }
    }
    return x;
}

double& Matrix::operator()(const int& i,
                           const int& j) {
    assert(m_size.inRange(i, j));
    return m_matrix[(i * getCols()) + j];
}

const double& Matrix::operator()(const int& i,
                                 const int& j) const {
    assert(m_size.inRange(i, j));
    return m_matrix[(i * getCols()) + j];
}

double& Matrix::at(const int& i,
                   const int& j) {
    assert(m_size.inRange(i, j));
    return m_matrix[(i * getCols()) + j];
}

const double& Matrix::at(const int& i,
                         const int& j) const {
    assert(m_size.inRange(i, j));
    return m_matrix[(i * getCols()) + j];
}

std::string Matrix::show() {
    std::string x = "{\n";
    int k = 0;
    for (int i = 0; i < getRows(); ++i) {
        x += "\t";
        for (int j = 0; j < getCols(); ++j) {
            std::string s = std::to_string(m_matrix[k]);
            x += s + "\t";
            k++;
        }
        x += "\n";
    }
    x += "}--"+ std::to_string(getRows()) + "X" + std::to_string(getCols());
    return x;
}

const Matrix Matrix::subMatrix(const int& a,
                               const int& b,
                               const ImageSize_t& size) const
{
    assert(m_size.inRange(a, b));
    assert(size < m_size);
    assert((a + m_size.nrows) <= getRows());
    assert((b + m_size.ncols) <= getCols());
    Matrix x(size, MTX_TYPE::ZEROS);
    for (int i = 0; i < size.nrows; ++i) {
        for (int j = 0; j < size.ncols; ++j) {
            x(i,j) = at(a + i, b + j);
        }
    }
    return x;
}

const double Matrix::sum() const {
    double c = 0;
    for (int i = 0; i < getRows() * getCols(); ++i) {
        c += m_matrix[i];
    }
    return c;
}

const Matrix Matrix::transpose() const {
    Matrix x(m_size.transpose(), MTX_TYPE::ZEROS);
    int k = 0;
    for (int i = 0; i < getRows(); ++i) {
        for (int j = 0; j < getCols(); ++j) {
            x(j,i) = m_matrix[k];
            k++;
        }
    }
    return x;
}

const Matrix Matrix::product(Matrix& m) const {
    assert(m.getRows() == getCols());
    Matrix x(ImageSize_t(getRows(),
                         m.getCols()),
             MTX_TYPE::ZEROS);
    for (int i = 0; i < x.getRows(); ++i) {
        for (int j = 0; j < x.getCols(); ++j) {
            double c = 0;
            for (int k = 0; k < getCols(); ++k) {
                c += at(i,k) * m(k, j);
            }
            x(i,j) = c;
        }
    }
    return x;
}

const Matrix Matrix::zeroPadding(const int& b) const {
    assert(b >= 0);
    Matrix x(ImageSize_t(getRows() + (2 * b),
                         getCols() + (2 * b)),
                         MTX_TYPE::ZEROS);
    for (int i = b; i < (getRows() + b); ++i) {
        for (int j = b; j < (getCols() + b); ++j) {
            x(i, j) = at(i - b, j - b);
        }
    }
    return x;
}

const Matrix Matrix::convolution(const Matrix& m) const {
    assert(isSquare());
    assert(m.isSquare());
    assert(m.getSize() < getSize());
    assert((m.getCols() % 2) != 0);
    Matrix y(ImageSize_t(getRows() - m.getRows() + 3,
                         getCols() - m.getCols() + 3),
             MTX_TYPE::ZEROS);
    Matrix x = zeroPadding(m.getRows() - 2);
    for (int i = 0; i < y.getRows(); ++i) {
        for (int j = 0; j < y.getCols(); ++j) {
            Matrix z = m * x.subMatrix(i, j, m.getSize());
            y(i,j) = z.sum();
        }
    }
    return y;
}

} /* base */
} /* capstone */
