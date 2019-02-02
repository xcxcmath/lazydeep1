//
// Created by bjk on 19. 1. 26.
//

#ifndef LAZYDEEP1_VARIABLE_HPP
#define LAZYDEEP1_VARIABLE_HPP

#include "Operand.hpp"

namespace lazy {
    template<typename T>
    class Variable : public Operand<T> {
    public:
        template<typename ...Types>
        explicit Variable(Types&& ...args): Operand<T>(){
            this->m_value = T(std::forward<Types>(args)...);
            this->m_optimizable = true;
        }

        // Anything about Copy/Move is inhibited
        LAZY_DELETED_FUNCTIONS(Variable, T);

        Variable& operator=(const T& val){
            this->resetValue();
            this->m_value.emplace(val);
            return *this;
        }

        const T& eval() override {
            return this->m_value.value();
        }
    };

    template<typename T, typename ...Types>
    decltype(auto) make_variable(Types&& ...args){
        return std::make_shared<Variable<T>>(std::forward<Types>(args)...);
    }

    template<typename T>
    decltype(auto) zero_matrix_variable(Index rows, Index cols){
        auto ret = make_variable<Matrix<T>>();
        auto m = Matrix<T>::Zero(rows, cols);

        *ret = m;
        return ret;
    }

    template<typename T>
    decltype(auto) random_normal_matrix_variable(Index rows, Index cols, T mean=0.0, T stddev=1.0) {
        auto ret = make_variable<Matrix<T>>();
        std::random_device rd;
        std::mt19937 gen{rd()};
        std::normal_distribution<T> nd(mean, stddev);

        Matrix<T> m(rows, cols);
        for(Index i = 0; i < rows; ++i) {
            for (Index j = 0; j < cols; ++j)
                m(i, j) = nd(gen);
        }

        *ret = m;
        return ret;
    }
}

#endif //LAZYDEEP1_VARIABLE_HPP
