//
// Created by bjk on 19. 1. 28.
//

#ifndef LAZYDEEP1_MATH_HPP
#define LAZYDEEP1_MATH_HPP

#include "Operator.hpp"

namespace lazy::math {
    template<typename T>
    [[nodiscard]] decltype(auto) exp
            (const T &t){
        LAZY_TYPEDEF_OPERATOR(T);

        return unaryExpr(t,
                         [](ScalarType f)->ScalarType{return std::exp(f);},
                         [](ScalarType f)->ScalarType{return std::exp(f);});
    }

    template<typename T>
    [[nodiscard]] decltype(auto) log
            (const T &t){
        LAZY_TYPEDEF_OPERATOR(T);

        return unaryExpr(t,
                         [](ScalarType f)->ScalarType{return std::log(f);},
                         [](ScalarType f)->ScalarType{return 1 / f;});
    }

    template<typename T>
    [[nodiscard]] decltype(auto) tanh
            (const T &t){
        LAZY_TYPEDEF_OPERATOR(T);

        return unaryExpr(t,
                         [](ScalarType f)->ScalarType{return std::tanh(f);},
                         [](ScalarType f)->ScalarType{return std::pow(std::cosh(f), ScalarType(-2));});
    }

    template<typename T>
    [[nodiscard]] decltype(auto) sigmoid
            (const T &t){
        LAZY_TYPEDEF_OPERATOR(T);

        return unaryExpr(t,
                         [](ScalarType f)->ScalarType{return 1 / (1 + std::exp(-f));},
                         [](ScalarType f)->ScalarType{return std::exp(-f) / std::pow(1 + std::exp(-f), ScalarType(2));});
    }
}

#endif //LAZYDEEP1_MATH_HPP
