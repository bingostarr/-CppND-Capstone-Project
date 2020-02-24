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
#include "defines.hpp"
#include "dataset.hpp"
#include "layer.hpp"

namespace capstone {
namespace base {

class Cnn {
public:
    explicit Cnn();
    ~Cnn() = default;
    void train(const DatasetImage& images,
               const DatasetLabel& labels);
    void test(const DatasetImage& images,
              const DatasetLabel& labels);
    std::string show();
private:
    std::vector<std::shared_ptr<Layer>> m_layers;
};

} /* base */
} /* capstone */
#endif /* INC_CNN_HPP_ */
