#include <omp.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>

#include "lazy/ops/NN.hpp"

#include "lazy/Variable.hpp"
#include "lazy/Placeholder.hpp"

#include "lazy/train/AdamOptimizer.hpp"
#include "lazy/train/MomentumOptimizer.hpp"

using namespace lazy;
using Mat = Matrix<float>;
using Byte = unsigned char;

constexpr Index TOTAL_SZ = 60'000;
constexpr Index TEST_SZ = 10'000;
constexpr Index PIXEL_SZ = 28 * 28;

const char TRAIN_IMG_FILE[] = "../examples/MNIST/train-images.idx3-ubyte";
const char TRAIN_LAB_FILE[] = "../examples/MNIST/train-labels.idx1-ubyte";
const char TEST_IMG_FILE[] = "../examples/MNIST/t10k-images.idx3-ubyte";
const char TEST_LAB_FILE[] = "../examples/MNIST/t10k-labels.idx1-ubyte";

Byte train_img[TOTAL_SZ][PIXEL_SZ];
Byte train_lab[TOTAL_SZ];
Byte test_img[TEST_SZ][PIXEL_SZ];
Byte test_lab[TEST_SZ];

bool load_data();

void build_input(Byte img[][PIXEL_SZ], Byte lab[], Mat& input, Mat& sol, Index batch, Index batch_sz);

int main() {
    /*
     * 3-layer NN example (MNIST)
     */

    std::cout << std::fixed;
    std::cout.precision(6);

    // Multi-threading
    // PLEASE CHANGE IT PROPERLY
    Eigen::setNbThreads(4);
    std::cout << Eigen::nbThreads() << " thread(s) ready\n";

    std::cout << "Loading files..";
    if(!load_data()){
        std::cout << "failed\n";
        return 0;
    }

    std::cout << "Finish\nInitializing model..";

    // hyper-parameters
    constexpr Index batch_sz = 100;
    constexpr Index total_batch = TOTAL_SZ / batch_sz;
    constexpr Index total_epoch = 15;
    constexpr Index test_batch = TEST_SZ / batch_sz;
    constexpr Index middle_layer = 256;

    /*
     * Construct NN Layers
     * 784 -> 256 -> 256 -> 10
     */

    // Placeholder : to insert input and label
    auto x = make_placeholder<Mat>(); // input
    auto t = make_placeholder<Mat>(); // solution label
    auto dropout_attr = make_placeholder<Mat>(); // for dropout

    // Variables : to optimize (i.e. Weights in this example)
    auto W1 = random_normal_matrix_variable<float>(middle_layer, PIXEL_SZ, 0.f, 0.01f);
    auto W2 = random_normal_matrix_variable<float>(middle_layer, middle_layer, 0.f, 0.01f);
    auto W3 = random_normal_matrix_variable<float>(10, middle_layer, 0.f, 0.01f);

    // Operands for middle layer
    // ReLU Activation Function is used
    auto wx1 = dot_product(W1, x);
    auto z1 = nn::relu(wx1);
    auto dz1 = nn::dropout(z1, dropout_attr);

    auto wx2 = dot_product(W2, dz1);
    auto z2 = nn::relu(wx2);
    auto dz2 = nn::dropout(z2, dropout_attr);

    // Operands for output layer
    auto wx3 = dot_product(W3, dz2);
    auto model = nn::softmax(wx3, nn::input_type::colwise);

    // cost function (or value) - smaller is better
    auto loss = nn::cross_entropy(model, t, nn::input_type::colwise, 1e-8f);

    // This Operand is used for testing, which counts correct answers
    auto corr = reduce_sum(equal(reduce_argmax(model, reduce_to::row),reduce_argmax(t, reduce_to::row)));

    std::cout << "Finish\n\n";

    /*
     * Optimize NN
     * minimize loss value using Adam Optimizing Algorithm
     */
    train::AdamOptimizer<Mat> opt(0.001f);
    auto training = opt.minimize(loss);

    Mat input(PIXEL_SZ, batch_sz);

    std::cout << "Training Start\n";
    auto train_start = std::chrono::system_clock::now();

    Placeholder<Mat>::applyPlaceholders({{dropout_attr, nn::dropout_attr_matrix<Mat>(0.5f, true)}});

    for(unsigned epoch = 0; epoch < total_epoch ; ++epoch){
        std::cout << "Epoch " << std::setw(2) << (epoch+1) << " : ";
        Mat total_cost = Mat::Zero(1, 1);

        for(Index batch = 0; batch < total_batch ; ++batch){
            Mat sol = Mat::Zero(10, batch_sz);

            // construct input and sol matrix
            build_input(train_img, train_lab, input, sol, batch, batch_sz);

            // training
            auto cost = training({{x, input},
                                  {t, sol}});
            total_cost = total_cost + cost;
        }

        // Average cost during one epoch
        auto avg_cost = total_cost(0) / total_batch;

        std::cout << avg_cost << '\n';
    }

    std::chrono::duration<double> train_elapsed = std::chrono::system_clock::now() - train_start;
    std::cout << "\nTraining Finish (Elapsed time : " << train_elapsed.count() << " sec)\n\n";

    /*
     * Test NN
     */
    std::cout << "Test Start..\n";
    Placeholder<Mat>::applyPlaceholders({{dropout_attr, nn::dropout_attr_matrix<Mat>(0.5f, false)}});

    int total_correct = 0;

    for(Index iter = 0; iter < test_batch; ++iter){
        // construct input
        Mat sol = Mat::Zero(10, batch_sz);
        build_input(test_img, test_lab, input, sol, iter, batch_sz);
        Placeholder<Mat>::applyPlaceholders({{x, input},
                                             {t, sol}});

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

bool load_bytes(const std::string &file, char* arr, long dummy_sz, long read_sz, const std::string& name){
    std::ifstream f(file, std::ios::binary);
    if(!f.is_open()){
        std::cout << "Failed to open " << name << '\n';
        return false;
    }

    f.seekg(dummy_sz, std::ios::beg);

    f.read(arr, read_sz);
    if(auto r = f.gcount(); r != read_sz){
        std::cout << "Failed to read data of " << name << " (read " << r << " bytes after dummies)\n";
        return false;
    }

    return true;
}

bool load_images(const std::string &file, Byte img[][PIXEL_SZ], long num, const std::string &name) {
    return load_bytes(file, reinterpret_cast<char*>(img), 16, num * PIXEL_SZ, name);
}

bool load_labels(const std::string &file, Byte *lab, long num, const std::string &name) {
    return load_bytes(file, reinterpret_cast<char*>(lab), 8, num, name);
}

bool load_data() {
    const bool openTrain = load_images(TRAIN_IMG_FILE, train_img, TOTAL_SZ, "train images") &&
                           load_labels(TRAIN_LAB_FILE, train_lab, TOTAL_SZ, "train labels");
    const bool openTest = load_images(TEST_IMG_FILE, test_img, TEST_SZ, "test images") &&
                          load_labels(TEST_LAB_FILE, test_lab, TEST_SZ, "test labels");

    return openTrain && openTest;
}


void build_input(Byte img[][PIXEL_SZ], Byte lab[], Mat& input, Mat& sol, Index batch, Index batch_sz){
    for(Index i = 0; i < batch_sz; ++i){
        for(Index p = 0; p < PIXEL_SZ; ++p){
            input(p, i) = (img[batch*batch_sz+i][p]/255.f);
        }
        sol(lab[batch*batch_sz+i], i) = 1.f;
    }
}