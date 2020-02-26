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

#include <iostream>

namespace capstone {
namespace base {

LayerConv::LayerConv(const LayerAttr_t& attr)
        : Layer(attr) {
    m_weights.clear();
    m_weightsCumul.clear();
    for (int i = 0; i < m_attr.outputSize.getNLayers(); ++i) {
        Matrix3d x(m_attr.weightSize, MTXTYPE::RANDN);
        m_weights.push_back(x);
        Matrix3d y(m_attr.weightSize);
        m_weightsCumul.push_back(y);
    }
}
void LayerConv::forward(Matrix3d& input,
                        Matrix3d& output) {
    output.zero();
    for (int k = 0; k < m_attr.outputSize.getNLayers(); ++k) {
        for (int i = 0; i < m_attr.outputSize.getSize(); ++i) {
            for (int j = 0; j < m_attr.outputSize.getSize(); ++j) {
                Matrix3d m = input.subMatrix3d(0, i, j, m_attr.weightSize) * m_weights[k];
                output(k, i, j) = m.sum();
            }
        }
    }
    m_cachedInput = input;
    m_cachedOutput = output;
}

void LayerConv::backward(Matrix3d& gradient) {
    m_gradient.zero();
    for (int k = 0; k < m_attr.outputSize.getNLayers(); ++k) {
        for (int i = 0; i < m_attr.outputSize.getSize(); ++i) {
            for (int j = 0; j < m_attr.outputSize.getSize(); ++j) {
                Matrix3d tmp(m_attr.inputSize);
                for (int kk = 0; kk < m_attr.weightSize.getNLayers(); ++kk) {
                    for (int ii = 0; ii < m_attr.weightSize.getSize(); ++ii) {
                        for (int jj = 0; jj < m_attr.weightSize.getSize(); ++jj) {
                            tmp(kk, i + ii, j + jj) = m_weights[k].at(kk, ii, jj);
                        }
                    }
                }
                m_gradient = m_gradient + (tmp * gradient(k, i, j));
            }
        }
    }

    std::vector<Matrix3d> weightsGrad;
    for (int i = 0; i < m_attr.outputSize.getNLayers(); ++i) {
        Matrix3d x(m_attr.weightSize);
        weightsGrad.push_back(x);
    }

    for (int k = 0; k < m_attr.outputSize.getNLayers(); ++k) {
        for (int i = 0; i < m_attr.outputSize.getSize(); ++i) {
            for (int j = 0; j < m_attr.outputSize.getSize(); ++j) {
                Matrix3d tmp(m_cachedInput.subMatrix3d(0, i, j, m_attr.weightSize));
                weightsGrad[k] = weightsGrad[k] + (tmp * gradient(k, i, j));
        }
      }
    }
    for (int k = 0; k < m_attr.outputSize.getNLayers(); ++k) {
        m_weightsCumul[k] = m_weightsCumul[k] + weightsGrad[k];
    }
}

void LayerConv::update(const double& diffLearningRate) {
    for (int k = 0; k < m_attr.outputSize.getNLayers(); ++k) {
        m_weights[k] = m_weights[k] - m_weightsCumul[k] * diffLearningRate;
    }
    reset();
}

void LayerConv::reset() {
    for (int i = 0; i < m_attr.outputSize.getNLayers(); ++i) {
        m_weightsCumul[i].zero();
    }
}

std::string LayerConv::show() {
    std::string x = LAYERTYPESTR[static_cast<int>(m_attr.type)];
    x += " {\n\tinput: " + m_attr.inputSize.show() + "\n";
    x += "\toutput: " + m_attr.outputSize.show() + "\n";
    x += "\tweights: " + std::to_string(m_attr.outputSize.getNLayers()) + "X" + m_attr.weightSize.show() + "\n";
    x += "}\n";
    return x;
}

void LayerPool::forward(Matrix3d& input,
                        Matrix3d& output) {
    output.zero();
    uint32_t K = m_attr.inputSize.getNLayers();
    uint32_t N = m_attr.inputSize.getSize();
    uint32_t n = m_attr.weightSize.getSize();
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i + n <= N; i += n) {
            for (int j = 0; j + n <= N; j += n) {
                output(k, i / n, j / n) =
                        input.at(k).subMatrix(i, j,
                                m_attr.weightSize.getImageSize()).max();
        }
      }
    }
    m_cachedInput = input;
    m_cachedOutput = output;
}

void LayerPool::backward(Matrix3d& gradient) {
    m_gradient.zero();
    uint32_t K = m_attr.inputSize.getNLayers();
    uint32_t N = m_attr.inputSize.getSize();
    uint32_t n = m_attr.weightSize.getSize();
    ImageSize_t imgSize = m_attr.weightSize.getImageSize();
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i + n <= N; i += n) {
            for (int j = 0; j + n <= N; j += n) {
                Matrix x(imgSize);
                Coords_t c = (m_cachedInput.at(k).subMatrix(i, j, imgSize)).getIndexMax();
                x(c.i, c.j) = gradient(k, i / n, j / n);
                for (int ii = 0; ii < n; ++ii) {
                    for (int jj = 0; jj < n; ++jj) {
                        m_gradient(k, i + ii, j + jj) += x(ii, jj);
                    }
                }
            }
        }
    }
}

std::string LayerPool::show() {
    std::string x = LAYERTYPESTR[static_cast<int>(m_attr.type)];
    x += " {\n\tinput: " + m_attr.inputSize.show() + "\n";
    x += "\toutput: " + m_attr.outputSize.show() + "\n";
    x += "\tpool: " + m_attr.weightSize.show() + "\n";
    x += "}\n";
    return x;
}

void LayerRelu::forward(Matrix3d& input,
                        Matrix3d& output) {
    output.zero();
    for (int k = 0; k < m_attr.outputSize.getNLayers(); ++k) {
        for (int i = 0; i < m_attr.outputSize.getSize(); ++i) {
            for (int j = 0; j < m_attr.outputSize.getSize(); ++j) {
                output(k, i, j) = std::max(input(k, i, j), output(k, i, j));
            }
        }
    }
    m_cachedInput = input;
    m_cachedOutput = output;
}

void LayerRelu::backward(Matrix3d& gradient) {
    m_gradient.zero();
    for (int k = 0; k < m_attr.inputSize.getNLayers(); ++k) {
        for (int i = 0; i < m_attr.inputSize.getSize(); ++i) {
            for (int j = 0; j < m_attr.inputSize.getSize(); ++j) {
                m_gradient(k, i, j) = (m_cachedInput(k, i, j) > 0) ? 1 : 0;
            }
        }
    }
    m_gradient = m_gradient * gradient;
}

std::string LayerRelu::show() {
    std::string x = LAYERTYPESTR[static_cast<int>(m_attr.type)];
    x += " {\n\tinput: " + m_attr.inputSize.show() + "\n";
    x += "\toutput: " + m_attr.outputSize.show() + "\n";
    x += "}\n";
    return x;
}

LayerFull::LayerFull(const LayerAttr_t& attr)
        : Layer(attr),
          m_gradientCumul(Matrix3d(m_attr.inputSize)),
          m_biases(Matrix3d(m_attr.outputSize)),
          m_biasesCumul(Matrix3d(m_attr.outputSize)) {
    m_weights.clear();
    m_weightsCumul.clear();
    for (int i = 0; i < m_attr.outputSize.getNLayers(); ++i) {
        Matrix3d x(m_attr.weightSize, MTXTYPE::RANDN);
        m_weights.push_back(x);
        Matrix3d y(m_attr.weightSize);
        m_weightsCumul.push_back(y);
    }
}

void LayerFull::forward(Matrix3d& input,
                        Matrix3d& output) {
    output.zero();
    for (int k = 0; k < m_attr.outputSize.getNLayers(); ++k) {
        Matrix3d s(m_weights[k] * input);
        double sumv = s.sum();
        output(k, 0, 0) = sumv + m_biases(k, 0, 0);
    }
    m_cachedInput = input;
    m_cachedOutput = output;
}

void LayerFull::backward(Matrix3d& gradient) {
    m_gradient.zero();
    for (int k = 0; k < m_attr.outputSize.getNLayers(); ++k) {
        m_gradient = m_gradient + m_weights[k] * gradient(k,0,0);
    }
    m_gradientCumul = m_gradientCumul + m_gradient;
    for (int k = 0; k < m_attr.outputSize.getNLayers(); ++k) {
        m_weightsCumul[k] = m_weightsCumul[k] + (m_cachedInput * gradient(k,0,0));
    }
    m_biasesCumul = m_biasesCumul + gradient;
}

void LayerFull::update(const double& diffLearningRate) {
    for (int k = 0; k < m_attr.outputSize.getNLayers(); ++k) {
        m_weights[k] = m_weights[k] - m_weightsCumul[k] * diffLearningRate;
    }
    m_biases = m_biases - m_biasesCumul * diffLearningRate;
    reset();
}

void LayerFull::reset() {
    m_gradientCumul.zero();
    m_biasesCumul.zero();
    for (int i = 0; i < m_attr.outputSize.getNLayers(); ++i) {
        m_weightsCumul[i].zero();
    }
}

std::string LayerFull::show() {
    std::string x = LAYERTYPESTR[static_cast<int>(m_attr.type)];
    x += " {\n\tinput: " + m_attr.inputSize.show() + "\n";
    x += "\toutput: " + m_attr.outputSize.show() + "\n";
    x += "\tweights: " + std::to_string(m_attr.outputSize.getNLayers()) + "X" + m_attr.weightSize.show() + "\n";
    x += "}\n";
    return x;
}

void LayerSmax::forward(Matrix3d& input,
                        Matrix3d& output) {
    output.zero();
    std::vector<double> v = input.vectorize();
    double maxv = *(std::max_element(v.begin(), v.end()));
    double sumv = 0;
    for (const double& d : v) {
        sumv += exp(d - maxv);
    }
    for (int i = 0; i < m_attr.outputSize.getNLayers(); ++i) {
        output(i,0,0) = exp(input(i,0,0) - maxv) / sumv;
    }
    m_cachedInput = input;
    m_cachedOutput = output;
}

void LayerSmax::backward(Matrix3d& gradient) {
    double sumv = 0;
    for (int i = 0; i < m_attr.outputSize.getNLayers(); ++i) {
        sumv += gradient(i,0,0) * m_cachedOutput(i,0,0);
    }
    m_gradient = (gradient - sumv) * m_cachedOutput;
}

std::string LayerSmax::show() {
    std::string x = LAYERTYPESTR[static_cast<int>(m_attr.type)];
    x += " {\n\tinput: " + m_attr.inputSize.show() + "\n";
    x += "\toutput: " + m_attr.outputSize.show() + "\n";
    x += "}\n";
    return x;
}

void LayerLoss::forward(Matrix3d& input,
                        Matrix3d& output) {
    m_loss = 0;
    for (int i = 0; i < m_attr.outputSize.getNLayers(); ++i) {
        m_loss += -1 * (output(i,0,0) * log(input(i,0,0)));
    }
    m_cachedInput = input;
    m_cachedOutput = output;
}

void LayerLoss::backward(Matrix3d& gradient) {
    m_gradient = (m_cachedOutput / m_cachedInput) * (-1);
}

std::string LayerLoss::show() {
    std::string x = LAYERTYPESTR[static_cast<int>(m_attr.type)];
    x += " {\n\tinput: " + m_attr.inputSize.show() + "\n";
    x += "\toutput: " + m_attr.outputSize.show() + "\n";
    x += "\tloss: " + std::to_string(m_loss) + "\n";
    x += "}\n";
    return x;
}

} /* base */
} /* capstone */
