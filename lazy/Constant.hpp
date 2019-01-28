//
// Created by bjk on 19. 1. 26.
//

#ifndef LAZYDEEP1_CONSTANT_HPP
#define LAZYDEEP1_CONSTANT_HPP

#include "Operand.hpp"

namespace lazy {
    template<typename T>
    class Constant : public Operand<T> {
    public:
        template<typename ...Types>
        explicit Constant(Types ...args): Operand<T>(){
            this->m_value.emplace(args...);
        }

        const T& eval() override {
            return this->m_value.value();
        }

        void setFunction(typename Operand<T>::Function f) override{

        }

        void setDFunction(typename Operand<T>::Function df) override{

        }

        void reset_value() override{

        }

        void reset_delta() override{

        }
    };
}

#endif //LAZYDEEP1_CONSTANT_HPP
