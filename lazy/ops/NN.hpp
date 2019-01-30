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
    [[nodiscard]] decltype(auto) relu
            (const T &t){
        LAZY_TYPEDEF_OPERATOR(T);

        return unaryExpr(t,
                [](ScalarType f)->ScalarType{return f > 0 ? f : 0;},
                [](ScalarType f)->ScalarType{return f > 0;});
    }

    template<typename T>
    [[nodiscard]] decltype(auto) softsign
            (const T &t){
        LAZY_TYPEDEF_OPERATOR(T);

        return unaryExpr(t,
                         [](ScalarType f)->ScalarType{return f / (std::abs(f) + 1);},
                         [](ScalarType f)->ScalarType{return std::pow(std::abs(f) + 1, ScalarType(-2));});
    }

    template<typename T>
    [[nodiscard]] decltype(auto) softmax
            (const T &t, input_type axis){
        LAZY_TYPEDEF_OPERATOR(T);

        auto ret = make_operand<ValueType>();
        ret->getPreOperand().insert({t});
        t->getPostOperand().insert({ret});

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

            t->getDF()[ret] = [ret](const PtrType& E) -> ValueType {
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
            t->getDF()[ret] = [ret](const PtrType& E) -> ValueType {
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
    [[nodiscard]] decltype(auto) cross_entropy
            (const T1 &t, const T2 &sol, input_type axis, T3 eps){
        LAZY_TYPEDEF_OPERATOR(T1);

        if(axis == input_type::colwise){
            return reduce_mean(reduce_sum(scalar_product(hadamard_product(sol, math::log(scalar_plus(t, eps))), -1.f), reduce_to::row), reduce_to::column);
        }

        // otherwise, rowwise input

        return reduce_mean(reduce_sum(scalar_product(hadamard_product(sol, math::log(scalar_plus(t, eps))), -1.f), reduce_to::column), reduce_to::row);
    }

    /*
     * Drop-out
     * t, attr -> mask (delta undefined)
     * t, mask -> ret (delta defined)
     */
    template<typename T> T dropout_attr_matrix
            (typename T::Scalar ratio, bool train){
        T ret(2, 1);
        ret << ratio, static_cast<typename T::Scalar>(train);
        return ret;
    }

    template<typename T1, typename T2>
    [[nodiscard]] decltype(auto) dropout
            (const T1 &t, const T2 &attr){
        LAZY_TYPEDEF_OPERATOR(T1);

        auto mask = make_operand<ValueType>();
        mask->getPreOperand().insert({t, attr});
        mask->setFunction([t, attr]() -> ValueType {
            const auto& ratio = attr->eval()(0);
            const auto& train = attr->eval()(1);
            if(train){
                std::random_device rd;
                std::mt19937 gen{rd()};
                std::bernoulli_distribution dist(ratio);
                return t->eval().unaryExpr([&dist, &gen](ScalarType)->ScalarType{
                    return dist(gen);
                });
            } else {
                return t->eval().unaryExpr([ratio](ScalarType)->ScalarType{
                    return ratio;
                });
            }
        });

        t->getPostOperand().insert({mask});
        // d(mask) / dt is undefined

        attr->getPostOperand().insert({mask});
        // d(mask) / d(attr) is undefined

        return hadamard_product(t, mask);
    }


}

#endif //LAZYDEEP1_NN_HPP
