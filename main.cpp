#include <omp.h>

#include <iostream>
#include <cstdio>
#include <iomanip>

#include "lazy/Operator.hpp"
#include "lazy/Variable.hpp"
#include "lazy/Placeholder.hpp"
#include "lazy/train/AdamOptimizer.hpp"

using namespace lazy;

constexpr int TOTAL_SZ = 60'000;
constexpr int TEST_SZ = 10'000;
constexpr int PIXEL_SZ = 28 * 28;

int images[TOTAL_SZ][PIXEL_SZ];
int labels[TOTAL_SZ];
int test_img[TEST_SZ][PIXEL_SZ];
int test_lab[TEST_SZ];

bool load_train(){
    FILE *img_in = fopen("../examples/MNIST/train-images.idx3-ubyte", "rb");
    FILE *lab_in = fopen("../examples/MNIST/train-labels.idx1-ubyte", "rb");

    if(img_in == nullptr || lab_in == nullptr) return false;

    for(int i = 0 ; i < 16 ; ++i) fgetc(img_in);
    for(int i = 0 ; i < 8 ; ++i) fgetc(lab_in);

    for(int idx = 0 ; idx < TOTAL_SZ ; ++idx){
        for(int p = 0 ; p < PIXEL_SZ ; ++p)
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

    for(int idx = 0 ; idx < TEST_SZ ; ++idx){
        for(int p = 0 ; p < PIXEL_SZ ; ++p)
            test_img[idx][p] = fgetc(img_in);
        test_lab[idx] = fgetc(lab_in);
    }

    fclose(img_in);
    fclose(lab_in);

    return true;
}

void build_train_input(Matrix<float>& input, Matrix<float>& sol, unsigned batch, unsigned batch_sz){
    for(int i = 0; i < batch_sz; ++i){
        for(int p = 0; p < PIXEL_SZ; ++p){
            input(p, i) = (images[batch*batch_sz+i][p]/255.f);
        }
        sol(labels[batch*batch_sz+i], i) = 1.f;
    }
}

void build_test_input(Matrix<float>& input, unsigned iter, unsigned test_sz){
    for(int i = 0; i < test_sz; ++i){
        for(int p = 0; p < PIXEL_SZ; ++p){
            input(p, i) = (test_img[iter*100+i][p]/255.f);
        }
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
    constexpr unsigned batch_sz = 100;
    constexpr unsigned total_batch = TOTAL_SZ / batch_sz;
    constexpr unsigned total_epoch = 15;
    constexpr unsigned middle_layer = 256;

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
    auto z1 = unaryExpr(wx1, [](float f)->float{return f > 0 ? f : 0;},
            [](float f)->float{return f > 0;});

    auto wx2 = dot_product(W2, z1);
    auto z2 = unaryExpr(wx2, [](float f)->float{return f > 0 ? f : 0;},
            [](float f)->float{return f > 0;});

    // Operands for output layer
    auto wx3 = dot_product(W3, z2);
    auto y = softmax(wx3);
    auto loss = cross_entropy(y, t); // scalar value; smaller is better

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

        for(unsigned batch = 0; batch < total_batch ; ++batch){
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
    std::cout << "Test Start\n";

    int total_correct = 0;
    input = Matrix<float>::Zero(PIXEL_SZ, 100);

    for(unsigned iter = 0; iter < 100; ++iter){
        std::cout << "Test " << std::setw(3) << (iter+1) << " ";

        int here = 0;

        // construct input matrix
        build_test_input(input, iter, 100);
        *x = input;

        // evaluate
        auto ans = y->eval();

        // check
        for(int i = 0; i < 100; ++i){
            Eigen::Index res, temp;
            ans.col(i).maxCoeff(&res, &temp);

            if((char)res == test_lab[iter*100+i]){
                ++here;
                ++total_correct;
            }
        }

        std::cout << std::setw(3) << here << "%\n";
    }

    std::cout << "\nTest Finish (" << total_correct << "/" << TEST_SZ << ")\n";

    return 0;
}