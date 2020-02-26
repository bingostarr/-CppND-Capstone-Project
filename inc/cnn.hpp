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
    static Cnn& getInstance()
    {
        static Cnn inst;
        return inst;
    }
    ~Cnn() = default;
    void train(DatasetImage& images,
               DatasetLabel& labels);
    TestResult_t test(DatasetImage& images,
                      DatasetLabel& labels,
                      const int& nImages);
    std::string show();
    std::string showAll();
private:
    Cnn();
    Cnn(const Cnn&) = delete;
    Cnn& operator=(const Cnn&) = delete;
    std::vector<std::shared_ptr<Layer>> m_layers;
    std::vector<Matrix3d> m_outputs;
};

} /* base */
} /* capstone */
#endif /* INC_CNN_HPP_ */
