//
// Created by bjk on 19. 1. 24.
//

#ifndef LAZYDEEP1_OPERATOR_HPP
#define LAZYDEEP1_OPERATOR_HPP

#include "Operand.hpp"

#define LAZY_ASSERT_TYPE_SAME(T1, T2) static_assert(std::is_same<T1, T2>::value, "lazy: Types are inconsistent")
#define LAZY_OPERATOR_TYPEDEF(T) \
    using _Operand = typename T::element_type; \
    using VecType = typename _Operand::PointerVec; \
    using ValueType = typename _Operand::ValueType; \
    using ScalarType = typename ValueType::Scalar;

namespace lazy {
    template<typename T1, typename T2>
    decltype(auto) colwise_plus
            (const T1 &t1, const T2 &vec){
        LAZY_OPERATOR_TYPEDEF(T1);

        auto ret = make_operand<ValueType>();
        ret->getPreOperand() = {t1, vec};
        t1->getPostOperand() = {ret};
        vec->getPostOperand() = {ret};

        ret->setFunction([](const VecType &v) -> ValueType{
            return v.at(0)->eval().colwise() + v.at(1)->eval().col(0);
        });
        t1->setDFunction([](const VecType &v) -> ValueType{
            return v.at(0)->diff();
        });
        vec->setDFunction([t1](const VecType &v) -> ValueType{
            return v.at(0)->diff().rowwise().sum();
        });

        return ret;
    }

    template<typename T1, typename T2>
    decltype(auto) rowwise_plus
            (const T1 &t1, const T2 &vec){
        LAZY_OPERATOR_TYPEDEF(T1);

        auto ret = make_operand<ValueType>();
        ret->getPreOperand() = {t1, vec};
        t1->getPostOperand() = {ret};
        vec->getPostOperand() = {ret};

        ret->setFunction([](const VecType &v) -> ValueType{
            return v.at(0)->eval().rowwise() + v.at(1)->eval().row(0);
        });
        t1->setDFunction([](const VecType &v) -> ValueType{
            return v.at(0)->diff();
        });
        vec->setDFunction([t1](const VecType &v) -> ValueType{
            return v.at(0)->diff().colwise().sum();
        });

        return ret;
    }

    template<typename T1, typename T2>
    decltype(auto) dot_product
            (const T1 &t1, const T2 &t2){
        LAZY_OPERATOR_TYPEDEF(T1);

        auto ret = make_operand<ValueType>();
        ret->getPreOperand() = {t1, t2};
        t1->getPostOperand() = {ret};
        t2->getPostOperand() = {ret};
        ret->setFunction([](const VecType &v) -> ValueType{
            return v.at(0)->eval() * v.at(1)->eval();
        });
        t1->setDFunction([t2](const VecType &v) -> ValueType{
            return v.at(0)->diff() * t2->eval().transpose();
        });
        t2->setDFunction([t1](const VecType &v) -> ValueType{
            return t1->eval().transpose() * v.at(0)->diff();
        });

        return ret;
    }

    template<typename T, typename F1, typename F2>
    decltype(auto) unaryExpr
            (const T &t, const F1 &func, const F2 &df){
        LAZY_OPERATOR_TYPEDEF(T);

        auto ret = make_operand<ValueType>();
        ret->getPreOperand() = {t};
        t->getPostOperand() = {ret};
        ret->setFunction([func](const VecType &v) -> ValueType{
            return v.at(0)->eval().unaryExpr(func);
        });
        t->setDFunction([t, df](const VecType &v) -> ValueType{
            return v.at(0)->diff().cwiseProduct(t->eval().unaryExpr(df));
        });

        return ret;
    }

    template<typename T>
    decltype(auto) softmax
            (const T &t){
        LAZY_OPERATOR_TYPEDEF(T);

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

    template<typename T1, typename T2>
    decltype(auto) cross_entropy
            (const T1 &t, const T2 &sol){
        LAZY_OPERATOR_TYPEDEF(T1);

        auto ret = make_operand<ValueType>();
        ret->getPreOperand() = {t, sol};
        t->getPostOperand() = {ret};
        sol->getPostOperand() = {ret};

        ret->setFunction([](const VecType &v) -> ValueType{
            ValueType loss(1, 1);
            const auto y = v.at(0)->eval();
            const auto ans = v.at(1)->eval();
            loss << (ans.cwiseProduct(y.unaryExpr(
                    [](ScalarType x){return std::log(x);})).sum() * -1 / y.cols());
            return loss;
        });

        t->setDFunction([t, sol](const VecType &v) -> ValueType{
            const auto tm = t->eval();
            return sol->eval().cwiseProduct(tm.unaryExpr([](ScalarType x){return -1/x;}))
                * v.at(0)->diff()(0, 0) / tm.cols();
        });
        sol->setDFunction([t](const VecType &v) -> ValueType{
            return t->eval().unaryExpr([](ScalarType x){return -std::log(x);}) * v.at(0)->diff()(0, 0);
        });

        return ret;
    }
}

#endif //LAZYDEEP1_OPERATOR_HPP
