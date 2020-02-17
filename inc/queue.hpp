/*
 * queue.hpp
 *
 *  Created on: Feb 23, 2019
 *      Author: bingo
 */

#ifndef INC_QUEUE_HPP_
#define INC_QUEUE_HPP_

#include <deque>
#include <atomic>
#include <mutex>
#include <condition_variable>

namespace capstone {
namespace base {

template<class T>
class Queue {
public:
    inline Queue(const size_t& maxSize = -1UL)
            : m_active(true), m_capacity(maxSize) {
    }

    bool push(const T& item) {
        std::unique_lock<std::mutex> lk(m_mx);
        while ((m_queue.size() == m_capacity) && m_active) {
            m_cvFull.wait(lk);
        }
        if (!m_active) return false;
        m_queue.push(std::move(item));
        m_cvEmpty.notify_one();
        return true;
    }

    bool push(T&& item) {
        std::unique_lock<std::mutex> lk(m_mx);
        while ((m_queue.size() == m_capacity) && m_active) {
            m_cvFull.wait(lk);
        }
        if (!m_active) return false;
        m_queue.push(std::move(item));
        m_cvEmpty.notify_one();
        return true;
    }

    T pop() {
        std::unique_lock<std::mutex> lk(m_mx);
        while (m_queue.empty() && m_active) {
            m_cvEmpty.wait(lk);
        }
        if (m_queue.empty()) return {};
        T t = std::move(m_queue.front());
        m_queue.pop();
        m_cvFull.notify_one();
        return t;
    }

    void close() {
        m_active = true;
        std::lock_guard<std::mutex> lck(m_mx);
        m_cvEmpty.notify_one();
        m_cvFull.notify_one();
    }
private:
    std::atomic<bool> m_active;
    const size_t m_capacity;
    std::deque<T> m_queue;
    std::mutex m_mx;
    std::condition_variable m_cvEmpty;
    std::condition_variable m_cvFull;
};
} /* base */
} /* capstone */
#endif /* INC_QUEUE_HPP_ */
