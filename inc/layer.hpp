/*
 * layer.hpp
 *
 *  Created on: Feb 17, 2020
 *      Author: bingo
 */

#ifndef INC_LAYER_HPP_
#define INC_LAYER_HPP_

#include <vector>
#include "defines.hpp"
#include "matrix3d.hpp"

namespace capstone {
namespace base {

class Layer {
public:
    explicit Layer(const LayerAttr_t& attr)
            : m_attr(attr),
              m_gradient(Matrix3d(m_attr.featureSize)),
              m_cachedInput(Matrix3d(m_attr.inputSize)),
              m_cachedOutput(Matrix3d(m_attr.outputSize)) {
    }
    virtual ~Layer() = default;
    inline const LayerAttr_t& getAttr() const {
        return m_attr;
    }
    inline Matrix3d& getGradient() {
        return m_gradient;
    }
    inline Matrix3d& getCachedInput() {
        return m_cachedInput;
    }
    inline Matrix3d& getCachedOutput() {
        return m_cachedOutput;
    }
    virtual void forward(Matrix3d& input,
                         Matrix3d& output) = 0;
    virtual void backward(Matrix3d& gradient) = 0;
protected:
    LayerAttr_t m_attr;
    Matrix3d m_gradient;
    Matrix3d m_cachedInput;
    Matrix3d m_cachedOutput;
};

#if 0
class LayerConv final : public Layer {
public:
    explicit LayerConv();
    virtual ~LayerConv() final = default;
    inline Matrix3d& getWeights() {
        return m_weights;
    }
    void forward(Matrix3d& input,
                 Matrix3d& output) final;
    void backward(Matrix3d& gradient) final;
private:
    const Matrix convolution(const Matrix& m) const;
    Matrix3d m_weights;
};
#endif

class LayerPool final : public Layer {
public:
    explicit LayerPool(const LayerAttr_t& attr,
                       const POOLTYPE& poolType = POOLTYPE::AVG)
            : Layer(attr),
              m_poolType(poolType) {
    }
    virtual ~LayerPool() final = default;
    inline const POOLTYPE& getPoolType() const {
        return m_poolType;
    }
    void forward(Matrix3d& input,
                 Matrix3d& output) final;
    void backward(Matrix3d& gradient) final;
private:
    POOLTYPE m_poolType;
};

class LayerRelu final : public Layer {
public:
    explicit LayerRelu(const LayerAttr_t& attr)
            : Layer(attr) {
    }
    virtual ~LayerRelu() final = default;
    void forward(Matrix3d& input,
                 Matrix3d& output) final;
    void backward(Matrix3d& gradient) final;
};

class LayerSmax final : public Layer {
public:
    explicit LayerSmax(const LayerAttr_t& attr)
            : Layer(attr) {
    }
    virtual ~LayerSmax() final = default;
    void forward(Matrix3d& input,
                 Matrix3d& output) final;
    void backward(Matrix3d& gradient) final;
};

class LayerLoss final : public Layer {
public:
    explicit LayerLoss(const LayerAttr_t& attr)
            : Layer(attr),
              m_loss(0) {
    }
    virtual ~LayerLoss() final = default;
    inline const double& getLoss() const {
        return m_loss;
    }
    void forward(Matrix3d& input,
                 Matrix3d& output) final;
    void backward(Matrix3d& gradient) final;
private:
    double m_loss;
};

} /* base */
} /* capstone */
#endif /* INC_LAYER_HPP_ */
