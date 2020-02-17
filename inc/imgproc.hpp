/*
 * imgproc.hpp
 *
 *  Created on: Feb 13, 2020
 *      Author: bingo
 */

#ifndef INC_IMGPROC_HPP_
#define INC_IMGPROC_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

namespace capstone {
namespace base {

inline void show(const uint32_t r,
                 const uint32_t c,
                 const std::vector<unsigned char> v,
                 const std::string l) {

    cv::Mat image(r, c, CV_8UC1, (void*)(&v[0]));

    if(!image.data)                              // Check for invalid input
    {
        return;
    }
    cv::namedWindow(l, cv::WINDOW_AUTOSIZE);// Create a window for display.
    cv::imshow(l, image);                   // Show our image inside it.
    cv::waitKey(0);                                          // Wait for a keystroke in the window
}
} /* base */
} /* capstone */
#endif /* INC_IMGPROC_HPP_ */
