//
// Created by bjk on 19. 1. 24.
//

#ifndef LAZYDEEP1_OPERATOR_HPP
#define LAZYDEEP1_OPERATOR_HPP

#include "../Operand.hpp"

#define LAZY_ASSERT_TYPE_SAME(T1, T2) static_assert(std::is_same<T1, T2>::value, "lazy: Types are inconsistent")
#define LAZY_TYPEDEF_OPERATOR(T) \
    using _Operand = typename T::element_type; \
    using PtrType = typename _Operand::Pointer; \
    using SetType = typename _Operand::PointerSet; \
    using MapType = typename _Operand::PointerMap; \
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
        ret->getPreOperand().insert({t});
        ret->setFunction([t, func]() -> ValueType{
            return t->eval().unaryExpr(func);
        });

        t->getPostOperand()[ret] = [t, ret, df](const PtrType& E) -> ValueType{
            return ret->diff(E).cwiseProduct(t->eval().unaryExpr(df));
        };

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
        ret->getPreOperand().insert({t1, vec});
        ret->setFunction([t1, vec]() -> ValueType {
            return t1->eval().colwise() + vec->eval().col(0);
        });

        t1->getPostOperand()[ret] = [ret](const PtrType& E) -> ValueType {
            return ret->diff(E);
        };
        vec->getPostOperand()[ret] = [ret](const PtrType& E) -> ValueType {
            return ret->diff(E).rowwise().sum();
        };

        return ret;
    }

    template<typename T1, typename T2>
    decltype(auto) rowwise_plus
            (const T1 &t1, const T2 &vec){
        LAZY_TYPEDEF_OPERATOR(T1);

        auto ret = make_operand<ValueType>();
        ret->getPreOperand().insert({t1, vec});
        ret->setFunction([t1, vec]() -> ValueType {
            return t1->eval().rowwise() + vec->eval().row(0);
        });

        t1->getPostOperand()[ret] = [ret](const PtrType& E) -> ValueType {
            return ret->diff(E);
        };
        vec->getPostOperand()[ret] = [ret](const PtrType& E) -> ValueType {
            return ret->diff(E).colwise().sum();
        };

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
        ret->getPreOperand().insert({t1, t2});
        ret->setFunction([t1, t2]() -> ValueType {
            return t1->eval() * t2->eval();
        });

        t1->getPostOperand()[ret] = [t2, ret](const PtrType& E) -> ValueType {
            return ret->diff(E) * t2->eval().transpose();
        };
        t2->getPostOperand()[ret] = [t1, ret](const PtrType& E) -> ValueType {
            return t1->eval().transpose() * ret->diff(E);
        };

        return ret;
    }

    template<typename T1, typename T2>
    decltype(auto) hadamard_product
            (const T1 &t1, const T2 &t2){
        LAZY_TYPEDEF_OPERATOR(T1);

        auto ret = make_operand<ValueType>();
        ret->getPreOperand().insert({t1, t2});
        ret->setFunction([t1, t2]() -> ValueType {
            return t1->eval().cwiseProduct(t2->eval());
        });

        t1->getPostOperand()[ret] = [t2, ret](const PtrType& E) -> ValueType {
            return ret->diff(E).cwiseProduct(t2->eval());
        };
        t2->getPostOperand()[ret] = [t1, ret](const PtrType& E) -> ValueType {
            return ret->diff(E).cwiseProduct(t1->eval());
        };

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
        ret->getPreOperand().insert({t});
        ret->setFunction([t, axis]() -> ValueType {
            if(axis == reduce_to::column)
                return t->eval().rowwise().sum();
            else if(axis == reduce_to::row)
                return t->eval().colwise().sum();

            ValueType m(1, 1);
            m << t->eval().sum();
            return m;
        });

        t->getPostOperand()[ret] = [t, ret, axis](const PtrType& E) -> ValueType {
            const auto rows = t->eval().rows();
            const auto cols = t->eval().cols();
            if(axis == reduce_to::column)
                return ret->diff(E) * ValueType::Ones(1, cols);
            else if(axis == reduce_to::row)
                return ValueType::Ones(rows, 1) * ret->diff(E);

            return ValueType::Constant(rows, cols, ret->diff(E)(0,0));
        };

        return ret;
    }

    template<typename T>
    decltype(auto) reduce_mean
            (const T& t, reduce_to axis){
        LAZY_TYPEDEF_OPERATOR(T);
        auto ret = make_operand<ValueType>();
        ret->getPreOperand().insert({t});
        ret->setFunction([t, axis]() -> ValueType {
            if(axis == reduce_to::column)
                return t->eval().rowwise().mean();
            else if(axis == reduce_to::row)
                return t->eval().colwise().mean();

            ValueType m(1, 1);
            m << t->eval().mean();
            return m;
        });

        t->getPostOperand()[ret] = [t, ret, axis](const PtrType& E) -> ValueType {
            const auto rows = t->eval().rows();
            const auto cols = t->eval().cols();
            if(axis == reduce_to::column)
                return ret->diff(E) * ValueType::Constant(1, cols, 1 / ScalarType(cols));
            else if(axis == reduce_to::row)
                return ValueType::Constant(rows, 1, 1 / ScalarType(rows)) * ret->diff(E);

            return ValueType::Constant(rows, cols, ret->diff(E)(0,0) / ScalarType(rows*cols));
        };

        return ret;
    }
}

#endif //LAZYDEEP1_OPERATOR_HPP
