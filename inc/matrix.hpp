/*
 * matrix.hpp
 *
 *  Created on: Feb 13, 2020
 *      Author: bingo
 */

#ifndef INC_MATRIX_HPP_
#define INC_MATRIX_HPP_

#include <memory>
#include <vector>

namespace capstone {
namespace base {

enum class MTX_TYPE {
    ZEROS,
    ONES,
    ID,
    RAND
};

typedef struct ImageSize {
    uint32_t nrows;
    uint32_t ncols;
    uint32_t npixels;
    ImageSize(const ImageSize& i)
            : nrows(i.nrows), ncols(i.ncols), npixels(nrows * ncols) { }
    ImageSize(ImageSize&& i)
            : nrows(i.nrows), ncols(i.ncols), npixels(nrows * ncols) { }
    ImageSize(const uint32_t& r, const uint32_t& c)
            : nrows(r), ncols(c), npixels(nrows * ncols) { }
    ImageSize(const uint32_t& n)
            : nrows(n), ncols(n), npixels(nrows * ncols) { }
    const std::string show() const {
        return std::to_string(nrows) + "X" + std::to_string(ncols);
    }
    const bool isSquare() const {
        return (nrows == ncols);
    }
    const bool inRange(const uint32_t& i,
                       const uint32_t& j) const {
        return (i >= 0 && i < nrows && j >= 0 && j < ncols);
    }
    inline const bool operator==(const ImageSize& rhs) const {
        return ((nrows == rhs.nrows) && (ncols == rhs.ncols));
    }
    inline const bool operator<(const ImageSize& rhs) const {
        return ((nrows < rhs.nrows) && (ncols < rhs.ncols));
    }
    const ImageSize transpose() const {
        ImageSize i(ncols, nrows);
        return i;
    }
} ImageSize_t;

class Matrix {
public:
    Matrix(const ImageSize_t& size,
           const MTX_TYPE& mtype = MTX_TYPE::ZEROS);
    Matrix(const ImageSize_t& size,
           const std::vector<double>& values = {});
    ~Matrix() = default;
    Matrix(const Matrix& m);
    Matrix(Matrix&& m);
    const Matrix operator=(const Matrix& m) const;
    const Matrix operator=(Matrix&& m) const;
    const Matrix operator+(const Matrix& m) const;
    const Matrix operator+(Matrix&& m) const;
    const Matrix operator-(const Matrix& m) const;
    const Matrix operator-(Matrix&& m) const;
    const Matrix operator*(const Matrix& m) const;
    const Matrix operator*(Matrix&& m) const;
    const Matrix operator/(const Matrix& m) const;
    const Matrix operator/(Matrix&& m) const;
    const Matrix operator*(const double& d) const;
    double& operator()(const int& i, const int& j);
    const double& operator()(const int& i, const int& j) const;
    double& at(const int& i, const int& j);
    const double& at(const int& i, const int& j) const;

    inline const ImageSize_t getSize() const {
        return m_size;
    }
    inline const uint32_t getRows() const {
        return m_size.nrows;
    }
    inline const uint32_t getCols() const {
        return m_size.ncols;
    }
    inline const bool isSquare() const {
        return (m_size.isSquare());
    }
    std::string show();
    const double sum() const;
    const Matrix subMatrix(const int& i,
                           const int& j,
                           const ImageSize_t& size) const;
    const Matrix transpose() const;
    const Matrix product(Matrix& m) const;
    const Matrix zeroPadding(const int& b = 1) const;
    const Matrix convolution(const Matrix& m) const;

private:
    ImageSize_t m_size;
    std::unique_ptr<double[]> m_matrix;
};
} /* base */
} /* capstone */
#endif /* INC_MATRIX_HPP_ */
