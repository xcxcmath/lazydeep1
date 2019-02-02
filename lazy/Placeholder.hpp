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

        // Anything about Copy/Move is inhibited
        LAZY_DELETED_FUNCTIONS(Placeholder, T);

        const T& diff(const typename Operand<T>::Pointer& E) override {
            const T& val = this->eval();
            return this->m_delta[E] = T::Zero(val.rows(), val.cols());
        }

        static void applyPlaceholders(const std::map<std::shared_ptr<Placeholder<T>>, T>& mp){
            for(const auto& [ptr, value] : mp){
                ptr->resetValue();
                ptr->m_value.emplace(value);
            }
        }
    };

    template<typename T>
    decltype(auto) make_placeholder(){
        return std::make_shared<Placeholder<T>>();
    }
}

#endif //LAZYDEEP1_PLACEHOLDER_HPP
