# CPPND Capstone Project
# Convolutional Neural Network for Classification of MNIST Handwritten Digit Database

This repo contains code for Capstone Project compiled as part of the [Udacity C++ Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213).

#### Goal
In this project, a Convolutional Neural Network (CNN) has been designed and built from scratch in C++, that attempts to learn and classify the [MNIST Handwritten Digit Database](http://yann.lecun.com/exdb/mnist/). The database contains various bitmaps of the handwritten versions of the digits: *'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'*.

The actual architecture of the Neural Network is based on the ["LeNet-5" publication](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf). More information can be found [here](http://yann.lecun.com/exdb/lenet/) and [here](https://engmrk.com/lenet-5-a-classic-cnn-architecture/). 

The structure and design of the source code for this project aspires to leverage key aspects of C++ programming language and showcase the concepts learned as part of the Nanodegree program. The project has been formulated to satisfy the project rubric specified in the Nanodegree program. The rubric is listed below.
#### Input
The input data for this application is the MNIST database, which was constructed using handwritten digits in black and white. More details can be found [here](http://yann.lecun.com/exdb/mnist/).
#### Performance
The efficacy of the CNN model is determined by the following key performance indicators:
* accuracy - captures if a digit/label has been classified correctly during training/validation/testing process; i.e., *is the classification wrong?*.
* loss - specifies the error encountered between classification and the actual label of the final output layer; i.e., *how wrong is the classification?*.

#### Code Directory Structure
* */inc* - C++ header files (**.hpp*)
* */src* - C++ source files (**.cpp*)
* */data* - training and test data and output files (**-ubyte*)

## Dependencies for Running Locally
* cmake >= 3.11
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
install v3.0 or greater.

## Build and Execute Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Copy data into build folder: `cp ../data/*ubyte .`
4. Run it: `./capstone_cnn`.

## Project Rubric

#### 1. README

| CRITERIA                                                     |           STATUS |
| ------------------------------------------------------------ | ---------------: |
| A README with instructions is included with the project      | DONE (README.md) |
| The README indicates which project is chosen.                | DONE (README.md) |
| The README includes information about each rubric point addressed. | DONE (README.md) |

#### 2. Compiling and Testing

| CRITERIA                             | STATUS |
| ------------------------------------ | -----: |
| The submission must compile and run. |   DONE |

#### 3. Loops, Functions, I/O

| CRITERIA                                                     |                            STATUS |
| ------------------------------------------------------------ | --------------------------------: |
| The project demonstrates an understanding of C++ functions and control structures. |  DONE (/src/dataset.cpp #78, #80) |
| The project reads data from a file and process the data, or the program writes data to a file. | DONE (/src/dataset.cpp #64, #172) |
| The project accepts user input and processes the input.      |                                NO |

#### 4. Object Oriented Programming

| CRITERIA                                                     |                 STATUS |
| ------------------------------------------------------------ | ---------------------: |
| The project uses Object Oriented Programming techniques.     |  DONE (/inc/layer.hpp) |
| Classes use appropriate access specifiers for class members. |  DONE (/inc/layer.hpp) |
| Class constructors utilize member initialization lists.      |  DONE (/inc/layer.hpp) |
| Classes abstract implementation details from their interfaces. |  DONE (/inc/layer.hpp) |
| Classes encapsulate behavior.                                |  DONE (/inc/layer.hpp) |
| Classes follow an appropriate inheritance hierarchy.         |  DONE (/inc/layer.hpp) |
| Overloaded functions allow the same function to operate on different parameters. | DONE (/inc/matrix.hpp) |
| Derived class functions override virtual base class functions. |  DONE (/inc/layer.hpp) |
| Templates generalize functions in the project.               |                     NO |

#### 4. Memory Management

| CRITERIA                                                     |                 STATUS |
| ------------------------------------------------------------ | ---------------------: |
| The project makes use of references in function declarations. | DONE (/inc/matrix.hpp) |
| The project uses destructors appropriately.                  | DONE (/inc/matrix.hpp) |
| The project uses scope / Resource Acquisition Is Initialization (RAII) where appropriate. | DONE (/inc/matrix.hpp) |
| The project follows the Rule of 5.                           | DONE (/inc/matrix.hpp) |
| The project uses move semantics to move data, instead of copying it, where possible. | DONE (/src/matrix.cpp) |
| The project uses smart pointers instead of raw pointers.     | DONE (/inc/matrix.hpp) |

#### 4. Concurrency

| CRITERIA                                     |                  STATUS |
| -------------------------------------------- | ----------------------: |
| The project uses multithreading.             | DONE (/src/dataset.cpp) |
| A promise and future is used in the project. |                      NO |
| A mutex or lock is used in the project.      | DONE (/inc/defines.cpp) |
| A condition variable is used in the project. | DONE (/inc/defines.cpp) |

## CNN Training, Validation and Test Results