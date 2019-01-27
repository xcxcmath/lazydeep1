//
// Created by bjk on 19. 1. 24.
//

#ifndef LAZYDEEP1_OPERATOR_HPP
#define LAZYDEEP1_OPERATOR_HPP

#include "../Operand.hpp"

#define LAZY_ASSERT_TYPE_SAME(T1, T2) static_assert(std::is_same<T1, T2>::value, "lazy: Types are inconsistent")
#define LAZY_TYPEDEF_OPERATOR(T) \
    using _Operand = typename T::element_type; \
    using VecType = typename _Operand::PointerVec; \
    using ValueType = typename _Operand::ValueType; \
    using ScalarType = typename ValueType::Scalar;

namespace lazy {

    /*
     * element-wise mapping
     */

    template<typename T, typename F1, typename F2>
    decltype(auto) unaryExpr
            (const T &t, const F1 &func, const F2 &df){
        LAZY_TYPEDEF_OPERATOR(T);

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

    /*
     * Plus
     */

    template<typename T1, typename T2>
    decltype(auto) colwise_plus
            (const T1 &t1, const T2 &vec){
        LAZY_TYPEDEF_OPERATOR(T1);

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
        LAZY_TYPEDEF_OPERATOR(T1);

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
    decltype(auto) scalar_plus
            (const T1 &t1, const T2 &constant){

        LAZY_TYPEDEF_OPERATOR(T1);
        LAZY_ASSERT_TYPE_SAME(ScalarType, T2);

        return unaryExpr(t1, [constant](ScalarType f)->ScalarType{return f + constant;},
                [constant](ScalarType f)->ScalarType{return 1;});
    }

    /*
     * Product
     */

    template<typename T1, typename T2>
    decltype(auto) scalar_product
            (const T1 &t1, const T2 &constant){

        LAZY_TYPEDEF_OPERATOR(T1);
        LAZY_ASSERT_TYPE_SAME(ScalarType, T2);

        return unaryExpr(t1, [constant](ScalarType f)->ScalarType{return f*constant;},
                         [constant](ScalarType f)->ScalarType{return constant;});
    }

    template<typename T1, typename T2>
    decltype(auto) dot_product
            (const T1 &t1, const T2 &t2){
        LAZY_TYPEDEF_OPERATOR(T1);

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

    template<typename T1, typename T2>
    decltype(auto) hadamard_product
            (const T1 &t1, const T2 &t2){
        LAZY_TYPEDEF_OPERATOR(T1);

        auto ret = make_operand<ValueType>();
        ret->getPreOperand() = {t1, t2};
        t1->getPostOperand() = {ret};
        t2->getPostOperand() = {ret};
        ret->setFunction([](const VecType &v) -> ValueType{
            return v.at(0)->eval().cwiseProduct(v.at(1)->eval());
        });
        t1->setDFunction([t2](const VecType &v) -> ValueType{
            return v.at(0)->diff().cwiseProduct(t2->eval());
        });
        t2->setDFunction([t1](const VecType &v) -> ValueType{
            return v.at(0)->diff().cwiseProduct(t1->eval());
        });

        return ret;
    }

    /*
     * reduction functions
     */

    enum class reduce_to {
        column,
        row,
        scalar
    };

    template<typename T>
    decltype(auto) reduce_sum
            (const T& t, reduce_to axis){
        LAZY_TYPEDEF_OPERATOR(T);

        auto ret = make_operand<ValueType>();
        ret->getPreOperand() = {t};
        t->getPostOperand() = {ret};

        if(axis == reduce_to::column){
            ret->setFunction([](const VecType &v) -> ValueType {
                return v.at(0)->eval().rowwise().sum();
            });
            t->setDFunction([t](const VecType &v) -> ValueType {
                const auto cols = t->eval().cols();
                return v.at(0)->diff() * ValueType::Ones(1, cols);
            });
        }
        else if(axis == reduce_to::row) {
            ret->setFunction([](const VecType &v) -> ValueType {
                return v.at(0)->eval().colwise().sum();
            });
            t->setDFunction([t](const VecType &v) -> ValueType {
                const auto rows = t->eval().rows();
                return ValueType::Ones(rows, 1) * v.at(0)->diff();
            });
        }
        else {
            ret->setFunction([](const VecType &v) -> ValueType {
                ValueType m(1, 1);
                m << v.at(0)->eval().sum();
                return m;
            });
            t->setDFunction([t](const VecType &v) -> ValueType {
                const auto rows = t->eval().rows();
                const auto cols = t->eval().cols();
                return ValueType::Constant(rows, cols, v.at(0)->diff()(0,0));
            });
        }

        return ret;
    }

    template<typename T>
    decltype(auto) reduce_mean
            (const T& t, reduce_to axis){
        LAZY_TYPEDEF_OPERATOR(T);

        auto ret = make_operand<ValueType>();
        ret->getPreOperand() = {t};
        t->getPostOperand() = {ret};
        if(axis == reduce_to::column){
            ret->setFunction([](const VecType &v) -> ValueType {
                return v.at(0)->eval().rowwise().mean();
            });
            t->setDFunction([t](const VecType &v) -> ValueType {
                const auto cols = t->eval().cols();
                return v.at(0)->diff() * ValueType::Constant(1, cols, 1 / ScalarType(cols));
            });
        }
        else if(axis == reduce_to::row) {
            ret->setFunction([](const VecType &v) -> ValueType {
                return v.at(0)->eval().colwise().mean();
            });
            t->setDFunction([t](const VecType &v) -> ValueType {
                const auto rows = t->eval().rows();
                return ValueType::Constant(rows, 1, 1 / ScalarType(rows)) * v.at(0)->diff();
            });
        }
        else {
            ret->setFunction([](const VecType &v) -> ValueType {
                ValueType m(1, 1);
                m << v.at(0)->eval().mean();
                return m;
            });
            t->setDFunction([t](const VecType &v) -> ValueType {
                const auto rows = t->eval().rows();
                const auto cols = t->eval().cols();
                return ValueType::Constant(rows, cols, v.at(0)->diff()(0,0) / ScalarType(rows*cols));
            });
        }

        return ret;
    }
}

#endif //LAZYDEEP1_OPERATOR_HPP
