//
// Created by bjk on 19. 1. 27.
//

#ifndef LAZYDEEP1_OPTIMIZER_HPP
#define LAZYDEEP1_OPTIMIZER_HPP

#include <set>
#include <map>
#include "../Variable.hpp"
#include "../Placeholder.hpp"

#define LAZY_TYPEDEF_OPTIMIZER \
using VariablePtrType = std::shared_ptr<Variable<T>>; \
using PlaceholderPtrType = std::shared_ptr<Placeholder<T>>; \
using OperandPtrType = typename Operand<T>::Pointer; \
using VariableMap = std::map<VariablePtrType, std::optional<T>>; \
using PlaceholderMap = std::map<PlaceholderPtrType, T>; \
using OptFunction = std::function<T(PlaceholderMap)>;

namespace lazy::train {
    template<typename T, typename Scalar = typename T::Scalar>
    class Optimizer {
    public:

        LAZY_TYPEDEF_OPTIMIZER

        explicit Optimizer(Scalar learning_rate): m_lr(learning_rate){}

        virtual OptFunction minimize(const OperandPtrType& target){
            return [this, target](PlaceholderMap mp) -> T{
                PlaceholderMap ph = std::move(mp);
                VariableMap grad = computeGradients(target, ph);
                T ret = target->eval();

                negateGradients(grad);
                applyGradients(grad);

                return ret;
            };
        };
        virtual OptFunction maximize(const OperandPtrType& target){
            return [this, target](PlaceholderMap mp) -> T{
                PlaceholderMap ph = std::move(mp);
                VariableMap grad = computeGradients(target, ph);
                T ret = target->eval();

                applyGradients(grad);

                return ret;
            };
        };

        virtual VariableMap computeGradients(const OperandPtrType& target, PlaceholderMap& ph){
            applyPlaceholders(ph);

            VariableMap var = searchVariables(target);
            for(auto& [ptr, value]: var){
                value = ptr->diff(target);
            }
            return var;
        }
        virtual void applyGradients(VariableMap& grad){
            for(auto& [ptr, value]: grad){
                *(ptr) = (ptr->eval()) + (value.value()) * m_lr;
            }
        }

    protected:
        Scalar m_lr;

        void negateGradients(VariableMap& grad){
            for(auto& [var, value]: grad){
                value = value.value() * static_cast<Scalar>(-1);
            }
        }
        void applyPlaceholders(PlaceholderMap& ph){
            for(auto& [ptr, value]: ph){
                *ptr = value;
            }
        }

        VariableMap searchVariables(const OperandPtrType& operand){
            VariableMap ret;
            for(const auto& pre: operand->getPreOperand()){
                const auto mp = searchVariables(pre);
                ret.insert(mp.begin(), mp.end());
            }
            if(operand->isOptimizable()) {
                ret.emplace(std::make_pair(
                        std::dynamic_pointer_cast<Variable<T>>(operand),
                        std::optional<T>(std::nullopt)));
            }

            return ret;
        }
    };
}

#endif //LAZYDEEP1_OPTIMIZER_HPP
