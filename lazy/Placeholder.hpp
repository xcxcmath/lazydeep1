//
// Created by bjk on 19. 1. 26.
//

#ifndef LAZYDEEP1_PLACEHOLDER_HPP
#define LAZYDEEP1_PLACEHOLDER_HPP

#include "Operand.hpp"

namespace lazy {
    template<typename T>
    class Placeholder : public Operand<T> {
    public:
        explicit Placeholder(): Operand<T>(){

        }

        Placeholder& operator=(const T& val){
            this->reset_value();
            this->m_value.emplace(val);
            return *this;
        }
    };

    template<typename T>
    decltype(auto) make_placeholder(){
        return std::make_shared<Placeholder<T>>();
    }
}

#endif //LAZYDEEP1_PLACEHOLDER_HPP
