#include <omp.h>

#include <iostream>
#include <iomanip>
#include <fstream>

#include "lazy/ops/NN.hpp"

#include "lazy/Variable.hpp"
#include "lazy/Placeholder.hpp"

#include "lazy/train/AdamOptimizer.hpp"

using namespace lazy;
using Mat = Matrix<float>;

constexpr Index TOTAL_SZ = 60'000;
constexpr Index TEST_SZ = 10'000;
constexpr Index PIXEL_SZ = 28 * 28;

unsigned char images[TOTAL_SZ][PIXEL_SZ];
unsigned char labels[TOTAL_SZ];
unsigned char test_img[TEST_SZ][PIXEL_SZ];
unsigned char test_lab[TEST_SZ];

bool load_train();
bool load_test();

void build_train_input(Mat& input, Mat& sol, Index batch, Index batch_sz);
void build_test_input(Mat& input, Mat& sol, Index iter, Index test_sz);

int main() {
    /*
     * 3-layer NN example (MNIST)
     */

    // Multi-threading
    // PLEASE CHANGE IT PROPERLY
    Eigen::setNbThreads(4);
    std::cout << Eigen::nbThreads() << " thread(s) ready\n";

    std::cout << "Loading files..";
    if(!load_train() || !load_test()){
        std::cout << "failed\n";
        return 0;
    }

    std::cout << "Finish\nInitializing model..";

    // hyper-parameters
    constexpr Index batch_sz = 100;
    constexpr Index total_batch = TOTAL_SZ / batch_sz;
    constexpr Index total_epoch = 15;
    constexpr Index middle_layer = 256;

    /*
     * Construct NN Layers
     * 784 -> 256 -> 256 -> 10
     */

    // Placeholder : to insert labels
    auto x = make_placeholder<Mat>(); // input
    auto t = make_placeholder<Mat>(); // solution label

    // Variables : to optimize (i.e. Weights in this example)
    auto W1 = random_normal_matrix_variable<float>(middle_layer, PIXEL_SZ, 0.f, 0.01f);
    auto W2 = random_normal_matrix_variable<float>(middle_layer, middle_layer, 0.f, 0.01f);
    auto W3 = random_normal_matrix_variable<float>(10, middle_layer, 0.f, 0.01f);

    // Operands for middle layer
    // ReLU Activation Function is used
    auto wx1 = dot_product(W1, x);
    auto z1 = nn::relu(wx1);

    auto wx2 = dot_product(W2, z1);
    auto z2 = nn::relu(wx2);

    // Operands for output layer
    auto wx3 = dot_product(W3, z2);
    auto y = nn::softmax(wx3, nn::input_type::colwise);

    // cost function (or value) - smaller is better
    auto loss = nn::cross_entropy(y, t, nn::input_type::colwise, 1e-8f);

    // This Operand is used for testing, which counts correct answers
    auto corr = reduce_sum(equal(reduce_argmax(y, reduce_to::row), reduce_argmax(t, reduce_to::row)));

    std::cout << "Finish\n\n";

    /*
     * Optimize NN
     * minimize loss value using Adam Optimizing Algorithm
     */
    train::AdamOptimizer<Mat> opt(0.001f);
    auto training = opt.minimize(loss);

    Mat input(PIXEL_SZ, batch_sz);

    std::cout << "Training Start\n";

    for(unsigned epoch = 0; epoch < total_epoch ; ++epoch){
        std::cout << "Epoch " << std::setw(2) << (epoch+1) << " : ";
        Mat total_cost = Mat::Zero(1, 1);

        for(Index batch = 0; batch < total_batch ; ++batch){
            Mat sol = Mat::Zero(10, batch_sz);

            // construct input and sol matrix
            build_train_input(input, sol, batch, batch_sz);

            // training
            auto cost = training({{x, input}, {t, sol}});
            total_cost = total_cost + cost;
        }

        // Average cost during one epoch
        total_cost /= total_batch;
        std::cout << total_cost << '\n';
    }

    std::cout << "\nTraining Finish\n\n";

    /*
     * Test NN
     */
    std::cout << "Test Start..\n";

    int total_correct = 0;
    input = Mat::Zero(PIXEL_SZ, 100);

    for(Index iter = 0; iter < 100; ++iter){
        // construct input
        Mat sol = Mat::Zero(10, 100);
        build_test_input(input, sol, iter, 100);
        Placeholder<Mat>::applyPlaceholders({{x, input}, {t, sol}});

        // evaluate
        const auto& res = corr->eval();
        total_correct += (int)res(0, 0);
    }

    std::cout << "Finish (" << total_correct << "/" << TEST_SZ << ")\n";

    return 0;
}

/*
 * MNIST Loader
 */

bool load_train(){
    std::ifstream img_in("../examples/MNIST/train-images.idx3-ubyte", std::ios::binary);
    std::ifstream lab_in("../examples/MNIST/train-labels.idx1-ubyte", std::ios::binary);

    if(!img_in.is_open() || !lab_in.is_open()) return false;

    char temp[16];
    img_in.read(temp, 16);
    if(auto r = img_in.gcount(); r != 16){
        std::cout << "something wrong with train temp " << r << '\n';
    }
    lab_in.read(temp, 8);
    if(auto r = lab_in.gcount(); r != 8){
        std::cout << "something wrong with train temp " << r << '\n';
    }

    img_in.read(reinterpret_cast<char*>(images), TOTAL_SZ * PIXEL_SZ);
    if(auto r = img_in.gcount(); r != TOTAL_SZ * PIXEL_SZ){
        std::cout << "something wrong with train image " << r << '\n';
        return false;
    }

    lab_in.read(reinterpret_cast<char*>(labels), TOTAL_SZ);
    if(auto r = lab_in.gcount(); r != TOTAL_SZ){
        std::cout << "something wrong with train label " << r << '\n';
        return false;
    }

    return true;
}

bool load_test(){
    std::ifstream img_in("../examples/MNIST/t10k-images.idx3-ubyte", std::ios::binary);
    std::ifstream lab_in("../examples/MNIST/t10k-labels.idx1-ubyte", std::ios::binary);

    if(!img_in.is_open() || !lab_in.is_open()) return false;

    char temp[16];
    img_in.read(temp, 16);
    if(auto r = img_in.gcount(); r != 16){
        std::cout << "something wrong with test temp " << r << '\n';
    }
    lab_in.read(temp, 8);
    if(auto r = lab_in.gcount(); r != 8){
        std::cout << "something wrong with test temp " << r << '\n';
    }

    img_in.read(reinterpret_cast<char*>(test_img), TEST_SZ * PIXEL_SZ);
    if(auto r = img_in.gcount(); r != TEST_SZ * PIXEL_SZ){
        std::cout << "something wrong with test image " << r << '\n';
        return false;
    }

    lab_in.read(reinterpret_cast<char*>(test_lab), TEST_SZ);
    if(auto r = lab_in.gcount(); r != TEST_SZ){
        std::cout << "something wrong with test label " << r << '\n';
        return false;
    }

    return true;
}

void build_train_input(Mat& input, Mat& sol, Index batch, Index batch_sz){
    for(Index i = 0; i < batch_sz; ++i){
        for(Index p = 0; p < PIXEL_SZ; ++p){
            input(p, i) = (images[batch*batch_sz+i][p]/255.f);
        }
        sol(labels[batch*batch_sz+i], i) = 1.f;
    }
}

void build_test_input(Mat& input, Mat& sol, Index iter, Index test_sz){
    for(Index i = 0; i < test_sz; ++i){
        for(Index p = 0; p < PIXEL_SZ; ++p){
            input(p, i) = (test_img[iter*test_sz+i][p]/255.f);
        }
        sol(test_lab[iter*test_sz+i], i) = 1.f;
    }
}