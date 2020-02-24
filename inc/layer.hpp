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
#include "matrix.hpp"

namespace capstone {
namespace base {

class Layer {
public:
    explicit Layer(const LayerAttr_t& attr)
            : m_attr(attr),
              m_gradient(Matrix3d(m_attr.inputSize)),
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
    virtual void update(const double& diffLearningRate) {
        //
    }
    virtual void reset() {
        //
    }
    virtual std::string show() = 0;
protected:
    LayerAttr_t m_attr;
    Matrix3d m_gradient;
    Matrix3d m_cachedInput;
    Matrix3d m_cachedOutput;
};

class LayerConv final : public Layer {
public:
    explicit LayerConv(const LayerAttr_t& attr);
    virtual ~LayerConv() final = default;
    void forward(Matrix3d& input,
                 Matrix3d& output) final;
    void backward(Matrix3d& gradient) final;
    void update(const double& diffLearningRate) final;
    void reset() final;
    std::string show() final;
private:
    Matrix3d m_gradientCumul;
    std::vector<Matrix3d> m_weights;
    std::vector<Matrix3d> m_weightsCumul;
};

class LayerPool final : public Layer {
public:
    explicit LayerPool(const LayerAttr_t& attr)
            : Layer(attr) {
    }
    virtual ~LayerPool() final = default;
    void forward(Matrix3d& input,
                 Matrix3d& output) final;
    void backward(Matrix3d& gradient) final;
    std::string show() final;
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
    std::string show() final;
};

class LayerFull final : public Layer {
public:
    explicit LayerFull(const LayerAttr_t& attr);
    virtual ~LayerFull() final = default;
    void forward(Matrix3d& input,
                 Matrix3d& output) final;
    void backward(Matrix3d& gradient) final;
    void update(const double& diffLearningRate) final;
    void reset() final;
    std::string show() final;
private:
    Matrix3d m_gradientCumul;
    Matrix3d m_biases;
    Matrix3d m_biasesCumul;
    std::vector<Matrix3d> m_weights;
    std::vector<Matrix3d> m_weightsCumul;
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
    std::string show() final;
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
    std::string show() final;
private:
    double m_loss;
};

} /* base */
} /* capstone */
#endif /* INC_LAYER_HPP_ */
