//
// Created by bjk on 19. 1. 27.
//

#ifndef LAZYDEEP1_ADAMOPTIMIZER_HPP
#define LAZYDEEP1_ADAMOPTIMIZER_HPP

#include "Optimizer.hpp"

namespace lazy::train {
    template<typename T, typename Scalar = typename T::Scalar>
    class AdamOptimizer : public Optimizer<T, Scalar> {
    public:

        LAZY_TYPEDEF_OPTIMIZER

        explicit AdamOptimizer(Scalar learning_rate,
                Scalar beta1 = Scalar(0.9),
                Scalar beta2 = Scalar(0.999),
                Scalar eps = Scalar(1e-8))
                :Optimizer<T, Scalar>(learning_rate), m_beta1(beta1),m_beta2(beta2),m_eps(eps){
            m_b1 = beta1;
            m_b2 = beta2;
        }

        using Optimizer<T, Scalar>::minimize;

        OptFunction minimize(const OperandPtrType& target, const VariableSet& var_list) override{
            return [this, target, var_list](PlaceholderMap mp) -> T{
                PlaceholderMap ph = std::move(mp);
                VariableMap grad = this->computeGradients(target, ph, var_list);
                T ret = target->eval();

                this->adjustMomentumAndGradients(grad);
                this->applyGradients(grad);

                return ret;
            };
        }

    protected:
        Scalar m_beta1, m_beta2;
        Scalar m_eps;
        Scalar m_b1, m_b2;

        VariableMap m_first;
        VariableMap m_second;

        void adjustMomentumAndGradients(VariableMap& grad){
            if(m_first.empty() || m_second.empty()){
                for(auto& [ptr, value]: grad){
                    m_first.emplace(std::make_pair(ptr, T::Zero(value.rows(), value.cols())));
                    m_second.emplace(std::make_pair(ptr, T::Zero(value.rows(), value.cols())));
                }
            }

            for(auto& [ptr, value]: grad){
                m_first.at(ptr) = m_first.at(ptr) * m_beta1 + value * (1-m_beta1);
                m_second.at(ptr) = m_second.at(ptr) * m_beta2
                        + value.unaryExpr([](Scalar f){return f*f;}) * (1-m_beta2);

                value = (m_first.at(ptr) / (1 - m_b1))
                        .cwiseProduct((m_second.at(ptr) / (1 - m_b2))
                        .unaryExpr([this](Scalar f){return std::pow(std::sqrt(f) + m_eps, Scalar(-1));}));
            }

            m_b1 *= m_beta1;
            m_b2 *= m_beta2;
        }
    };
}

#endif //LAZYDEEP1_ADAMOPTIMIZER_HPP
