#include <omp.h>

#include <iostream>
#include <cstdio>
#include <iomanip>

#include "lazy/ops/NN.hpp"

#include "lazy/Variable.hpp"
#include "lazy/Placeholder.hpp"

#include "lazy/train/AdamOptimizer.hpp"

using namespace lazy;

constexpr Index TOTAL_SZ = 60'000;
constexpr Index TEST_SZ = 10'000;
constexpr Index PIXEL_SZ = 28 * 28;

unsigned char images[TOTAL_SZ][PIXEL_SZ];
unsigned char labels[TOTAL_SZ];
unsigned char test_img[TEST_SZ][PIXEL_SZ];
unsigned char test_lab[TEST_SZ];

bool load_train(){
    FILE *img_in = fopen("../examples/MNIST/train-images.idx3-ubyte", "rb");
    FILE *lab_in = fopen("../examples/MNIST/train-labels.idx1-ubyte", "rb");

    if(img_in == nullptr || lab_in == nullptr) return false;

    for(int i = 0 ; i < 16 ; ++i) fgetc(img_in);
    for(int i = 0 ; i < 8 ; ++i) fgetc(lab_in);

    for(Index idx = 0 ; idx < TOTAL_SZ ; ++idx){
        for(Index p = 0 ; p < PIXEL_SZ ; ++p)
            images[idx][p] = fgetc(img_in);
        labels[idx] = fgetc(lab_in);
    }

    fclose(img_in);
    fclose(lab_in);

    return true;
}

bool load_test(){
    FILE *img_in = fopen("../examples/MNIST/t10k-images.idx3-ubyte", "rb");
    FILE *lab_in = fopen("../examples/MNIST/t10k-labels.idx1-ubyte", "rb");

    if(img_in == nullptr || lab_in == nullptr) return false;

    for(int i = 0 ; i < 16 ; ++i) fgetc(img_in);
    for(int i = 0 ; i < 8 ; ++i) fgetc(lab_in);

    for(Index idx = 0 ; idx < TEST_SZ ; ++idx){
        for(Index p = 0 ; p < PIXEL_SZ ; ++p)
            test_img[idx][p] = fgetc(img_in);
        test_lab[idx] = fgetc(lab_in);
    }

    fclose(img_in);
    fclose(lab_in);

    return true;
}

void build_train_input(Matrix<float>& input, Matrix<float>& sol, Index batch, Index batch_sz){
    for(Index i = 0; i < batch_sz; ++i){
        for(Index p = 0; p < PIXEL_SZ; ++p){
            input(p, i) = (images[batch*batch_sz+i][p]/255.f);
        }
        sol(labels[batch*batch_sz+i], i) = 1.f;
    }
}

void build_test_input(Matrix<float>& input, Matrix<float>& sol, Index iter, Index test_sz){
    for(Index i = 0; i < test_sz; ++i){
        for(Index p = 0; p < PIXEL_SZ; ++p){
            input(p, i) = (test_img[iter*test_sz+i][p]/255.f);
        }
        sol(test_lab[iter*test_sz+i], i) = 1.f;
    }
}

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

    std::cout << "Finish\nInitializing network..";

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
    auto x = make_placeholder<Matrix<float>>(); // input
    auto t = make_placeholder<Matrix<float>>(); // solution label

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
    train::AdamOptimizer<Matrix<float>> opt(0.001f);
    auto training = opt.minimize(loss);

    Matrix<float> input(PIXEL_SZ, batch_sz);

    std::cout << "Training Start\n";

    for(unsigned epoch = 0; epoch < total_epoch ; ++epoch){
        std::cout << "Epoch " << std::setw(2) << (epoch+1) << " : ";
        Matrix<float> total_cost = Matrix<float>::Zero(1, 1);

        for(Index batch = 0; batch < total_batch ; ++batch){
            Matrix<float> sol = Matrix<float>::Zero(10, batch_sz);

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
    input = Matrix<float>::Zero(PIXEL_SZ, 100);

    for(Index iter = 0; iter < 100; ++iter){
        // construct input
        Matrix<float> sol = Matrix<float>::Zero(10, 100);
        build_test_input(input, sol, iter, 100);
        *x = input;
        *t = sol;

        // evaluate
        const auto& res = corr->eval();
        total_correct += (int)res(0, 0);
    }

    std::cout << "Finish (" << total_correct << "/" << TEST_SZ << ")\n";

    return 0;
}