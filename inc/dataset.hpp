/*
 * dataset.hpp
 *
 *  Created on: Feb 13, 2020
 *      Author: bingo
 */

#ifndef INC_DATASET_HPP_
#define INC_DATASET_HPP_

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>
#include "matrix.hpp"

namespace capstone {
namespace base {

typedef struct Synch {
    std::mutex mx;
    std::condition_variable cv;
    bool ready = {false};
    void set() {
        std::unique_lock<std::mutex> lock(mx);
        ready = true;
        cv.notify_one();
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mx);
        if (!ready) {
            cv.wait(lock, [this] {return (this->ready);});
        }
    }
} Synch_t;

class Dataset {
public:
    explicit Dataset(const std::string& filename);
    virtual ~Dataset() = default;
    virtual void init() = 0;
    std::string show();
    void wait();
    inline const std::string& getFileName() const {
        return m_filename;
    }
    inline const uint32_t& getMagicNumber() const {
        return m_magicNumber;
    }
    inline const uint32_t& getNImages() const {
        return m_nImages;
    }

protected:
    Synch_t m_synch;
    std::string m_filename;
    uint32_t m_magicNumber;
    uint32_t m_nImages;
};

class DatasetImage final : public Dataset {
public:
    explicit DatasetImage(const std::string& filename);
    ~DatasetImage() final = default;
    void init() final;
    std::string showIndex(const int& index);
    Matrix& operator()(const int& i) {
        return m_data[i];
    }
    const Matrix& operator()(const int& i) const {
        return m_data[i];
    }

private:
    std:: vector<Matrix> m_data;
};

class DatasetLabel final : public Dataset {
public:
    explicit DatasetLabel(const std::string& filename);
    ~DatasetLabel() final = default;
    void init() final;
    std::string showIndex(const int& index);
    inline double& operator()(const int& i) {
        return m_data[i];
    }
    const double& operator()(const int& i) const {
        return m_data[i];
    }

private:
    std:: vector<double> m_data;
};

} /* base */
} /* capstone */
#endif /* INC_DATASET_HPP_ */
