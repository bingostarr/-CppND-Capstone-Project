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
#include "defines.hpp"

namespace capstone {
namespace base {

class Matrix {
public:
    Matrix(const ImageSize_t& size,
           const MTXTYPE& mtype = MTXTYPE::ZEROS);
    Matrix(const ImageSize_t& size,
           const std::vector<double>& values);
    Matrix(const Matrix& m);
    Matrix(Matrix&& m);
    ~Matrix() = default;
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
    const Matrix operator+(const double& d) const;
    const Matrix operator-(const double& d) const;
    const Matrix operator*(const double& d) const;
    const Matrix operator/(const double& d) const;
    double& operator()(const int& i, const int& j);
    const double& operator()(const int& i, const int& j) const;
    double& at(const int& i, const int& j);
    const double& at(const int& i, const int& j) const;

    inline const ImageSize_t getImageSize() const {
        return m_size;
    }
    inline const uint32_t getSize() const {
        return m_size.getSize();
    }
    inline const uint32_t getNPixels() const {
        return m_size.nPixels();
    }
    std::string show();
    const double sum() const;
    const Coords_t getIndexMax() const;
    const double& max() const;
    const Coords_t getIndexMin() const;
    const double& min() const;
    const Matrix subMatrix(const int& a,
                           const int& b,
                           const ImageSize_t& size) const;
    const Matrix transpose() const;
    const Matrix product(Matrix& m) const;
    const std::vector<double> vectorize() const;
    void zero();

private:
    ImageSize_t m_size;
    std::unique_ptr<double[]> m_matrix;
};
} /* base */
} /* capstone */
#endif /* INC_MATRIX_HPP_ */
