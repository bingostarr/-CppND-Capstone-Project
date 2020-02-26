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
    Matrix3d(const CubeSize_t& size,
             const MTXTYPE& mtxType = MTXTYPE::ZEROS);
    Matrix3d(const Matrix3d& m);
    Matrix3d(Matrix3d&& m);
    Matrix3d(const Matrix& m);
    ~Matrix3d() = default;
    Matrix3d operator=(const Matrix3d& m);
    Matrix3d operator=(Matrix3d&& m);
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
    double& at(const int& k,
               const int& i,
               const int& j);
    const double& at(const int& k,
                     const int& i,
                     const int& j) const;

    inline const CubeSize_t& getCubeSize() const {
        return m_size;
    }
    inline CubeSize_t&& moveCubeSize() {
        return std::move(m_size);
    }
    inline const ImageSize_t& getImageSize() const {
        return m_size.getImageSize();
    }
    inline const uint32_t& getNLayers() const {
        return m_size.getNLayers();
    }
    inline const uint32_t& getSize() const {
        return m_size.getSize();
    }
    inline std::vector<Matrix>&& moveMatrix3d() {
        return std::move(m_matrix3d);
    }
    std::string show();
    const double sum() const;
    const Matrix3d subMatrix3d(const int& c,
                               const int& a,
                               const int& b,
                               const CubeSize_t& size) const;
    void fillSubMatrix3d(const int& c,
                         const int& a,
                         const int& b,
                         const Matrix3d& m);
    const std::vector<double> vectorize() const;
    const std::vector<double> vectorizeColWise() const;
    void zero();
private:
    CubeSize_t m_size;
    std::vector<Matrix> m_matrix3d;
};
} /* base */
} /* capstone */
#endif /* INC_MATRIX3D_HPP_ */
