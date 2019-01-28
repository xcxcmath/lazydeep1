//
// Created by bjk on 19. 1. 28.
//

#ifndef LAZYDEEP1_NN_HPP
#define LAZYDEEP1_NN_HPP

#include "Math.hpp"

namespace lazy::nn {

    enum class input_type {
        colwise,
        rowwise
    };

    /*
     * Activation Functions
     */
    template<typename T>
    decltype(auto) relu
            (const T &t){
        LAZY_TYPEDEF_OPERATOR(T);

        return unaryExpr(t,
                [](ScalarType f)->ScalarType{return f > 0 ? f : 0;},
                [](ScalarType f)->ScalarType{return f > 0;});
    }

    template<typename T>
    decltype(auto) softmax
            (const T &t, input_type axis){
        LAZY_TYPEDEF_OPERATOR(T);

        auto ret = make_operand<ValueType>();
        ret->getPreOperand().insert({t});


        if(axis == input_type::colwise) {
            ret->setFunction([t]() -> ValueType {
                const auto& m = t->eval();
                const auto ex = (m.rowwise() - m.colwise().maxCoeff())
                        .unaryExpr([](ScalarType x) { return std::exp(x); });
                const auto ex_sum = ex.colwise().sum()
                        .unaryExpr([](ScalarType x) { return std::pow(x, static_cast<ScalarType>(-1)); })
                        .row(0).asDiagonal();
                return ex * ex_sum;
            });
            t->getPostOperand()[ret] = [ret](const PtrType& E) -> ValueType {
                const auto& m = ret->eval();
                const auto& d = ret->diff(E);
                const auto colwise_prod_sum = m.cwiseProduct(d).colwise().sum();
                const auto d_minus_sum = d.rowwise() - colwise_prod_sum;
                return d_minus_sum.cwiseProduct(m);
            };
        }
        else {
            ret->setFunction([t]() -> ValueType {
                const auto& m = t->eval();
                const auto ex = (m.colwise() - m.rowwise().maxCoeff())
                        .unaryExpr([](ScalarType x) { return std::exp(x); });
                const auto ex_sum = ex.rowwise().sum()
                        .unaryExpr([](ScalarType x) { return std::pow(x, static_cast<ScalarType>(-1)); })
                        .col(0).asDiagonal();
                return ex_sum * ex;
            });
            t->getPostOperand()[ret] = [ret](const PtrType& E) -> ValueType {
                const auto& m = ret->eval();
                const auto& d = ret->diff(E);
                const auto rowwise_prod_sum = m.cwiseProduct(d).rowwise().sum();
                const auto d_minus_sum = d.colwise() - rowwise_prod_sum;
                return d_minus_sum.cwiseProduct(m);
            };
        }

        return ret;
    }

    template<typename T1, typename T2, typename T3>
    decltype(auto) cross_entropy
            (const T1 &t, const T2 &sol, input_type axis, T3 eps){
        LAZY_TYPEDEF_OPERATOR(T1);

        if(axis == input_type::colwise){
            return reduce_mean(reduce_sum(scalar_product(hadamard_product(sol, math::log(scalar_plus(t, eps))), -1.f), reduce_to::row), reduce_to::column);
        }

        // otherwise, rowwise input

        return reduce_mean(reduce_sum(scalar_product(hadamard_product(sol, math::log(scalar_plus(t, eps))), -1.f), reduce_to::column), reduce_to::row);
    }

}

#endif //LAZYDEEP1_NN_HPP
