/*
 * matrix3d.hpp
 *
 *  Created on: Feb 22, 2020
 *      Author: bingo
 */

#ifndef INC_MATRIX3D_HPP_
#define INC_MATRIX3D_HPP_

#include <vector>
#include "defines.hpp"
#include "matrix.hpp"

namespace capstone {
namespace base {

class Matrix3d {
public:
    Matrix3d(const CubeSize_t& size);
    Matrix3d(const Matrix3d& m);
    Matrix3d(Matrix3d&& m);
    ~Matrix3d() = default;
    const Matrix3d operator=(const Matrix3d& m) const;
    const Matrix3d operator=(Matrix3d&& m) const;
    const Matrix3d operator+(const Matrix3d& m) const;
    const Matrix3d operator+(Matrix3d&& m) const;
    const Matrix3d operator-(const Matrix3d& m) const;
    const Matrix3d operator-(Matrix3d&& m) const;
    const Matrix3d operator*(const Matrix3d& m) const;
    const Matrix3d operator*(Matrix3d&& m) const;
    const Matrix3d operator/(const Matrix3d& m) const;
    const Matrix3d operator/(Matrix3d&& m) const;
    const Matrix3d operator+(const double& d) const;
    const Matrix3d operator-(const double& d) const;
    const Matrix3d operator*(const double& d) const;
    const Matrix3d operator/(const double& d) const;
    double& operator()(const int& k,
                       const int& i,
                       const int& j);
    const double& operator()(const int& k,
                             const int& i,
                             const int& j) const;
    Matrix& at(const int& k);
    const Matrix& at(const int& k) const;

    inline const CubeSize_t getSize() const {
        return m_size;
    }
    inline const ImageSize_t getImageSize() const {
        return m_size.getImageSize();
    }
    inline const uint32_t getNLayers() const {
        return m_size.getNLayers();
    }
    std::string show();
    const std::vector<double> vectorize() const;

private:
    CubeSize_t m_size;
    std::vector<Matrix> m_matrix3d;
};
} /* base */
} /* capstone */
#endif /* INC_MATRIX3D_HPP_ */