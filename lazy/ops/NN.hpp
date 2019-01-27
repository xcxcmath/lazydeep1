//
// Created by bjk on 19. 1. 28.
//

#ifndef LAZYDEEP1_NN_HPP
#define LAZYDEEP1_NN_HPP

#include "Math.hpp"

namespace lazy::nn {

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
    decltype(auto) colwise_softmax
            (const T &t){
        LAZY_TYPEDEF_OPERATOR(T);

        auto ret = make_operand<ValueType>();
        ret->getPreOperand() = {t};
        t->getPostOperand() = {ret};

        ret->setFunction([](const VecType &v) -> ValueType{
            const auto m = v.at(0)->eval();
            const auto ex = (m.rowwise() - m.colwise().maxCoeff())
                    .unaryExpr([](ScalarType x){return std::exp(x);});
            const auto ex_sum = ex.colwise().sum()
                    .unaryExpr([](ScalarType x){return std::pow(x, static_cast<ScalarType>(-1));})
                    .row(0).asDiagonal();
            return ex * ex_sum;
        });
        t->setDFunction([ret](const VecType &v) -> ValueType{
            const auto m = ret->eval();
            const auto d = v.at(0)->diff();
            const auto colwise_prod_sum = m.cwiseProduct(d).colwise().sum();
            const auto d_minus_sum = d.rowwise() - colwise_prod_sum;
            return d_minus_sum.cwiseProduct(m);
        });

        return ret;
    }

    template<typename T1, typename T2, typename T3>
    decltype(auto) col_batch_cross_entropy
            (const T1 &t, const T2 &sol, T3 eps){
        LAZY_TYPEDEF_OPERATOR(T1);

        return reduce_mean(reduce_sum(scalar_product(hadamard_product(sol, math::log(scalar_plus(t, 1e-8f))), -1.f), reduce_to::row), reduce_to::column);

        // The source code below is older version of cross entropy

        /*
        auto ret = make_operand<ValueType>();
        ret->getPreOperand() = {t, sol};
        t->getPostOperand() = {ret};
        sol->getPostOperand() = {ret};

        ret->setFunction([](const VecType &v) -> ValueType{
            ValueType loss(1, 1);
            const auto y = v.at(0)->eval();
            const auto ans = v.at(1)->eval();
            loss << (ans.cwiseProduct(y.unaryExpr(
                    [](ScalarType x){return std::log(x + ScalarType(1e-8));})).sum() * -1 / y.cols());
            return loss;
        });

        t->setDFunction([t, sol](const VecType &v) -> ValueType{
            const auto tm = t->eval();
            return sol->eval().cwiseProduct(tm.unaryExpr([](ScalarType x){return -1/(x + ScalarType(1e-8));}))
                   * v.at(0)->diff()(0, 0) / tm.cols();
        });
        sol->setDFunction([t](const VecType &v) -> ValueType{
            return t->eval().unaryExpr([](ScalarType x){return -std::log(x + ScalarType(1e-8));}) * v.at(0)->diff()(0, 0);
        });

        return ret;
         */
    }

    template<typename T1, typename T2, typename T3>
    decltype(auto) row_batch_cross_entropy
            (const T1 &t, const T2 &sol, T3 eps){
        LAZY_TYPEDEF_OPERATOR(T1);

        return reduce_mean(reduce_sum(scalar_product(hadamard_product(sol, math::log(scalar_plus(t, 1e-8f))), -1.f), reduce_to::column), reduce_to::row);

        // The source code below is older version of cross entropy

        /*
        auto ret = make_operand<ValueType>();
        ret->getPreOperand() = {t, sol};
        t->getPostOperand() = {ret};
        sol->getPostOperand() = {ret};

        ret->setFunction([](const VecType &v) -> ValueType{
            ValueType loss(1, 1);
            const auto y = v.at(0)->eval();
            const auto ans = v.at(1)->eval();
            loss << (ans.cwiseProduct(y.unaryExpr(
                    [](ScalarType x){return std::log(x + ScalarType(1e-8));})).sum() * -1 / y.cols());
            return loss;
        });

        t->setDFunction([t, sol](const VecType &v) -> ValueType{
            const auto tm = t->eval();
            return sol->eval().cwiseProduct(tm.unaryExpr([](ScalarType x){return -1/(x + ScalarType(1e-8));}))
                   * v.at(0)->diff()(0, 0) / tm.cols();
        });
        sol->setDFunction([t](const VecType &v) -> ValueType{
            return t->eval().unaryExpr([](ScalarType x){return -std::log(x + ScalarType(1e-8));}) * v.at(0)->diff()(0, 0);
        });

        return ret;
         */
    }

}

#endif //LAZYDEEP1_NN_HPP
