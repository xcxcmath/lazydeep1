//
// Created by bjk on 19. 1. 28.
//

#ifndef LAZYDEEP1_MATH_HPP
#define LAZYDEEP1_MATH_HPP

#include "Operator.hpp"

namespace lazy::math {
    template<typename T>
    decltype(auto) exp
            (const T &t){
        LAZY_TYPEDEF_OPERATOR(T);

        return unaryExpr(t,
                         [](ScalarType f)->ScalarType{return std::exp(f);},
                         [](ScalarType f)->ScalarType{return std::exp(f);});
    }

    template<typename T>
    decltype(auto) log
            (const T &t){
        LAZY_TYPEDEF_OPERATOR(T);

        return unaryExpr(t,
                         [](ScalarType f)->ScalarType{return std::log(f);},
                         [](ScalarType f)->ScalarType{return 1 / f;});
    }
}

#endif //LAZYDEEP1_MATH_HPP
