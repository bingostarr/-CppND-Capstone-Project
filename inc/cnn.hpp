/*
 * cnn.hpp
 *
 *  Created on: Feb 23, 2020
 *      Author: bingo
 */

#ifndef INC_CNN_HPP_
#define INC_CNN_HPP_

#include <vector>
#include <memory>
#include "dataset.hpp"
#include "layer.hpp"
#include "matrix3d.hpp"

namespace capstone {
namespace base {

class Cnn {
public:
    explicit Cnn();
    ~Cnn() = default;
    void train(DatasetImage& images,
               DatasetLabel& labels);
    void test(DatasetImage& images,
              DatasetLabel& labels);
    std::string show();
private:
    std::vector<std::shared_ptr<Layer>> m_layers;
    std::vector<Matrix3d> m_outputs;
};

} /* base */
} /* capstone */
#endif /* INC_CNN_HPP_ */
