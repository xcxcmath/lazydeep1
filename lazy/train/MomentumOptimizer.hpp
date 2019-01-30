//
// Created by bjk on 19. 1. 30.
//

#ifndef LAZYDEEP1_MOMENTUMOPTIMIZER_HPP
#define LAZYDEEP1_MOMENTUMOPTIMIZER_HPP

#include "Optimizer.hpp"

namespace lazy::train {
    template<typename T, typename Scalar = typename T::Scalar>
    class MomentumOptimizer : public Optimizer<T, Scalar> {
    public:

        LAZY_TYPEDEF_OPTIMIZER

        explicit MomentumOptimizer(Scalar learning_rate,
                               Scalar momentum = Scalar(0.9),
                               bool use_nesterov = false)
                :Optimizer<T, Scalar>(learning_rate), m_momentum(momentum), m_nag(use_nesterov){

        }

        OptFunction minimize(const OperandPtrType& target) override{
            VariableSet var_list = this->searchVariables(target);
            return minimize(target, var_list);
        };

        OptFunction minimize(const OperandPtrType& target, const VariableSet& var_list) override{
            return [this, target, var_list](PlaceholderMap mp) -> T{
                PlaceholderMap ph = std::move(mp);
                VariableMap grad = this->computeGradients(target, ph, var_list);
                T ret = target->eval();

                if(m_nag) {
                    this->applyMomentum(grad);
                    grad = this->computeGradients(target, ph, var_list);
                }
                this->adjustMomentumWith(grad);
                this->applyGradients(grad);

                return ret;
            };
        }

    protected:
        Scalar m_momentum;
        bool m_nag;

        VariableMap m_accumulation;

        void initMomentum(VariableMap& grad){
            for(auto& [ptr, value]: grad){
                m_accumulation.emplace(std::make_pair(ptr, T::Zero(value.rows(), value.cols())));
            }
        }

        void adjustMomentumWith(VariableMap& grad){
            if(m_accumulation.empty()){
                initMomentum(grad);
            }

            for(auto& [ptr, value]: grad){
                value = m_accumulation.at(ptr) = m_accumulation.at(ptr) * m_momentum + value;
            }
        }

        void applyMomentum(VariableMap& grad){
            if(m_accumulation.empty()){
                initMomentum(grad);
            }
            for(auto& [ptr, value]: grad){
                *(ptr) = (ptr->eval()) + m_accumulation.at(ptr) * -(this->m_lr) * m_momentum;
            }
        }
    };
}

#endif //LAZYDEEP1_MOMENTUMOPTIMIZER_HPP
