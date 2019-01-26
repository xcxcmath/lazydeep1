//
// Created by bjk on 19. 1. 27.
//

#ifndef LAZYDEEP1_ADAMOPTIMIZER_H
#define LAZYDEEP1_ADAMOPTIMIZER_H

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

        OptFunction minimize(const OperandPtrType& target) override{
            return [this, target](PlaceholderMap mp) -> T {
                PlaceholderMap ph = std::move(mp);
                VariableMap grad = this->computeGradients(target, ph);
                T ret = target->eval();

                this->adjustGradients(grad);
                this->negateGradients(grad);
                this->applyGradients(grad);

                return ret;
            };
        };
        OptFunction maximize(const OperandPtrType& target) override {
            return [this, target](PlaceholderMap mp) -> T {
                PlaceholderMap ph = std::move(mp);
                VariableMap grad = this->computeGradients(target, ph);
                T ret = target->eval();

                this->adjustGradients(grad);
                this->applyGradients(grad);

                return ret;
            };
        };

    protected:
        Scalar m_beta1, m_beta2;
        Scalar m_eps;
        Scalar m_b1, m_b2;

        VariableMap m_first;
        VariableMap m_second;

        void adjustGradients(VariableMap& grad){
            if(m_first.empty() || m_second.empty()){
                for(auto& [ptr, value]: grad){
                    const auto here = value->eval();
                    m_first.emplace(std::make_pair(ptr, T::Zero(here.rows(), here.cols())));
                    m_second.emplace(std::make_pair(ptr, T::Zero(here.rows(), here.cols())));
                }
            }

            for(auto& [ptr, value]: grad){
                const auto here = value->eval();
                m_first.at(ptr) = m_first.at(ptr).value() * m_beta1 + here * (1-m_beta1);
                m_second.at(ptr) = m_second.at(ptr).value() * m_beta2
                        + here.unaryExpr([](Scalar f){return f*f;}) * (1-m_beta2);

                value = (m_first.at(ptr).value() / (1 - m_b1))
                        .cwiseProduct((m_second.at(ptr).value() / (1 - m_b2))
                        .unaryExpr([this](Scalar f){return std::pow(std::sqrt(f) + m_eps, Scalar(-1));}));
            }

            m_b1 *= m_beta1;
            m_b2 *= m_beta2;
        }
    };
}

#endif //LAZYDEEP1_ADAMOPTIMIZER_H
