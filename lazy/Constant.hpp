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
        explicit Constant(Types&& ...args): Operand<T>(){
            this->m_value.emplace(std::forward<Types>(args)...);
        }

        // Anything about Copy/Move is inhibited
        LAZY_DELETED_FUNCTIONS(Constant, T);

        const T& eval() override {
            return this->m_value.value();
        }

        void setFunction(typename Operand<T>::Function) override {
            // do nothing
        }

        void reset_value() override {
            for (const auto &ptr: this->m_post) ptr->resetValue();
        }
    };
}

#endif //LAZYDEEP1_CONSTANT_HPP
