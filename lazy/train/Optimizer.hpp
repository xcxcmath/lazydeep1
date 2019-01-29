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
using VariableSet = std::set<VariablePtrType>; \
using VariableMap = std::map<VariablePtrType, T>; \
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
                VariableSet var_list = searchVariables(target);
                VariableMap grad = computeGradients(target, ph, var_list);
                T ret = target->eval();

                applyGradients(grad);

                return ret;
            };
        };

        virtual OptFunction minimize(const OperandPtrType& target, const VariableSet& var_list){
            return [this, target, var_list](PlaceholderMap mp) -> T{
                PlaceholderMap ph = std::move(mp);
                VariableMap grad = computeGradients(target, ph, var_list);
                T ret = target->eval();

                applyGradients(grad);

                return ret;
            };
        }

        virtual VariableMap computeGradients(const OperandPtrType& target, PlaceholderMap& ph, const VariableSet& var_list){
            applyPlaceholders(ph);

            VariableMap var_map;
            for(auto& ptr: var_list){
                var_map[ptr] = ptr->diff(target);
            }
            return var_map;
        }
        virtual void applyGradients(VariableMap& grad){
            for(auto& [ptr, value]: grad){
                *(ptr) = (ptr->eval()) + (value) * -m_lr;
            }
        }

    protected:
        Scalar m_lr;

        void applyPlaceholders(PlaceholderMap& ph){
            for(auto& [ptr, value]: ph){
                *ptr = value;
            }
        }

        VariableSet searchVariables(const OperandPtrType& operand){
            VariableSet ret;
            for(const auto& pre: operand->getPreOperand()){
                const auto mp = searchVariables(pre);
                ret.insert(mp.begin(), mp.end());
            }
            if(operand->isOptimizable()) {
                ret.insert(std::dynamic_pointer_cast<Variable<T>>(operand));
            }

            return ret;
        }
    };
}

#endif //LAZYDEEP1_OPTIMIZER_HPP
