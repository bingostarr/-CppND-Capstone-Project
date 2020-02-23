/*
 * layer.cpp
 *
 *  Created on: Feb 13, 2020
 *      Author: bingo
 */

#include "layer.hpp"
#include <cassert>
#include <math.h>
#include <algorithm>

namespace capstone {
namespace base {

//#if 0
//void Matrix::convolution(const Matrix& input,
//                         Matrix& output) const {
//    assert(isSquare());
//    assert(kernel.isSquare());
//    assert(kernel.getSize() < getSize());
//    assert((kernel.getCols() % 2) != 0);
//    Matrix y(ImageSize_t(getRows() - kernel.getRows() + 2 * p + 1),
//             MTXTYPE::ZEROS);
//    Matrix x = padding(p);
//    for (int i = 0; i < y.getRows(); ++i) {
//        for (int j = 0; j < y.getCols(); ++j) {
//            Matrix z = kernel * x.subMatrix(i, j, kernel.getSize());
//            y(i,j) = z.sum();
//        }
//    }
//    return y;
//}
//
//#endif
//

void LayerPool::forward(Matrix3d& input,
                        Matrix3d& output) {
    assert(input.getSize() == m_cachedInput.getSize());
    assert(output.getSize() == m_cachedOutput.getSize());
    assert(input.getNLayers() == output.getNLayers());
    assert(output.getNLayers() != 0);
    assert((input.getImageSize().getSize() % m_attr.featureSize.getImageSize().getSize()) == 0);
    uint32_t N = m_attr.inputSize.getImageSize().getSize();
    ImageSize_t imgSize = m_attr.featureSize.getImageSize();
    uint32_t K = m_attr.featureSize.getNLayers();
    uint32_t n = imgSize.getSize();
    output = Matrix3d(m_attr.outputSize);
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i + n <= N; i += n) {
            for (int j = 0; j + n <= N; j += n) {
                output(k, i/n, j/n) =
                        input.at(k).subMatrix(i, j, imgSize).max();
        }
      }
    }
    m_cachedInput = input;
    m_cachedOutput = output;
}

void LayerPool::backward(Matrix3d& gradient) {
    assert(gradient.getSize() == m_cachedOutput.getSize());
    m_gradient = Matrix3d(m_attr.featureSize);
    uint32_t N = m_attr.inputSize.getImageSize().getSize();
    ImageSize_t imgSize = m_attr.featureSize.getImageSize();
    uint32_t K = m_attr.featureSize.getNLayers();
    uint32_t n = imgSize.getSize();
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i + n <= N; i += n) {
            for (int j = 0; j + n <= N; j += n) {
                Matrix x(imgSize);
                Coords_t c = m_cachedInput.at(k).subMatrix(i, j, imgSize).getIndexMax();
                x(c.i, c.j) = gradient(k, i, j);
                for (int ii = 0; ii < n; ++ii) {
                    for (int jj = 0; jj < imgSize.getSize(); ++jj) {
                        m_gradient(k, i + ii, j + jj) += x(ii, jj);
                    }
                }
            }
        }
    }
}

void LayerSmax::forward(Matrix3d& input,
                        Matrix3d& output) {
    assert(input.getSize() == m_cachedInput.getSize());
    assert(output.getSize() == m_cachedOutput.getSize());
    assert(input.getNLayers() == output.getNLayers());
    assert(input.getNLayers() != 0);
    assert(input.getSize().getImageSize().getSize() == 1);
    assert(output.getSize().getImageSize().getSize() == 1);
    output = Matrix3d(m_attr.outputSize);

    std::vector<double> v = input.vectorize();
    double maxv = *(std::max_element(v.begin(), v.end()));
    double sumv = 0;
    for (const double& d : v) {
        sumv += exp(d - maxv);
    }
    for (int i = 0; i < output.getNLayers(); ++i) {
        output(i,0,0) = exp(v[i] - maxv)/sumv;
    }
//    for (const Matrix& m : input)
    m_cachedInput = input;
    m_cachedOutput = output;
}

void LayerSmax::backward(Matrix3d& gradient) {
    assert(gradient.getNLayers() == m_cachedOutput.getNLayers());
    assert(gradient.getNLayers() == m_gradient.getNLayers());
    double sumv = 0;
    for (int i = 0; i < gradient.getNLayers(); ++i) {
        sumv += gradient(i,0,0) * m_cachedOutput(i,0,0);
    }
    m_gradient = (gradient - sumv) * m_cachedOutput;
}

void LayerLoss::forward(Matrix3d& input,
                        Matrix3d& output) {
    assert(input.getNLayers() == output.getNLayers());
    assert(input.getNLayers() != 0);
    m_loss = 0;
    for (int i = 0; i < input.getNLayers(); ++i) {
        m_loss += -1 * (output(i,0,0) * log(input(i,0,0)));
    }
    m_cachedInput = input;
    m_cachedOutput = output;
}

void LayerLoss::backward(Matrix3d& gradient) {
    assert(m_gradient.getNLayers() == m_cachedInput.getNLayers());
    assert(m_gradient.getNLayers() == m_cachedOutput.getNLayers());
    m_gradient = (m_cachedOutput / m_cachedInput) * (-1);
}

} /* base */
} /* capstone */
