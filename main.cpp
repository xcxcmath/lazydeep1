#include <iostream>
#include <iomanip>
#include "lazy/Operator.hpp"
#include "lazy/Variable.hpp"
#include "lazy/Placeholder.hpp"
#include "lazy/train/AdamOptimizer.hpp"

using namespace lazy;

int main() {
    /*
     * 2-layer NN example (Rock scissors paper)
     */
    constexpr unsigned ROCK = 0;
    constexpr unsigned SCISSORS = 1;
    constexpr unsigned PAPER = 2;

    unsigned res[3];
    res[ROCK] = PAPER;      // PAPER    wins ROCK
    res[SCISSORS] = ROCK;   // ROCK     wins SCISSORS
    res[PAPER] = SCISSORS;  // SCISSORS wins PAPER

    // Our goal is to make NN win on the RSP games
    // when its opposite's hand is known

    // hyper-parameters
    constexpr unsigned middle = 20;
    constexpr unsigned batch_sz = 5;
    constexpr unsigned batch_iter = 30;

    /*
     * Construct NN Layers
     */

    // Placeholder : to insert labels
    auto x = make_placeholder<Matrix<float>>(); // input
    auto t = make_placeholder<Matrix<float>>(); // solution label

    // Variables : to optimize (i.e. Weights in this example)
    auto W1 = random_normal_matrix_variable<float>(middle, 3, 0, std::sqrt(2.f/3.f));
    auto W2 = random_normal_matrix_variable<float>(3, middle, 0, std::sqrt(2.f/middle));

    // Operands for middle layer
    auto wx1 = dot_product(W1, x);
    auto z1 = unaryExpr(wx1, [](float f)->float{return f > 0 ? f : 0;},
            [](float f)->float{return f > 0;});

    // Operands for output layer
    auto wx2 = dot_product(W2, z1);
    auto y = softmax(wx2);
    auto loss = cross_entropy(y, t); // scalar value; smaller is better

    /*
     * Optimize NN
     * minimize loss value using Adam Optimizing Algorithm
     */
    train::AdamOptimizer<Matrix<float>> opt(0.01);
    auto training = opt.minimize(loss);

    std::random_device rd;
    std::uniform_int_distribution<int> random_dist(0, 2);

    for(unsigned i = 0; i < batch_iter; ++i){

        Matrix<float> input = Matrix<float>::Zero(3, batch_sz);
        Matrix<float> ans = Matrix<float>::Zero(3, batch_sz);

        for(unsigned j = 0; j < batch_sz; ++j){
            int test_in = random_dist(rd);

            // ONE-HOT vectors
            input(test_in, j) = 1.f;
            ans(res[test_in], j) = 1.f;
        }

        // train to derive (ans) from (input)
        auto cost = training({{x, input}, {t, ans}});

        if((i+1) % (batch_iter/10) == 0)
            std::cout << "Batch " << std::setw(4) << (i+1) << " : " << cost << std::endl;
    }

    /*
     * Test NN
     */
    for(unsigned i = 0; i < 3; ++i){
        Matrix<float> input = Matrix<float>::Zero(3, 1);
        input(i, 0) = 1.;
        *x = input;

        auto ans = y->eval();
        Eigen::Index r, c; ans.maxCoeff(&r, &c);

        std::cout << "Test " << i << " : ";
        if(r == res[i]){
            std::cout << "  Correct (";
        } else {
            std::cout << "Incorrect (";
        }
        std::cout << std::setw(6) << (ans(res[i], 0) * 100) << " %)\n";
    }

    return 0;
}