//
// Created by bjk on 19. 1. 24.
//

#ifndef LAZYDEEP1_OPERAND_HPP
#define LAZYDEEP1_OPERAND_HPP

#include <Eigen>
#include <memory>
#include <algorithm>
#include <functional>
#include <optional>
#include <vector>

namespace lazy {
    template<typename T>
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    template<typename T>
    class Operand {
    public:
        using ValueType = T;
        using Pointer = std::shared_ptr<Operand<T>>;
        using PointerVec = std::vector<Pointer>;
        using Function = std::optional<std::function<T(const PointerVec&)>>;

        explicit Operand()
        : m_pre(), m_post(),
        m_f(std::nullopt), m_df(std::nullopt),
        m_value(std::nullopt), m_delta(std::nullopt),
        m_optimizable(false){
            resetDFunction();
        }

        virtual T eval(){
            if(m_value.has_value())
                return m_value.value();

            m_value = m_f.value()(m_pre);
            return m_value.value();
        }

        virtual T diff(){
            return m_delta.has_value() ? m_delta.value() : m_delta.emplace(m_df.value()(m_post));
        }

        PointerVec& getPreOperand() {
            return m_pre;
        }
        const PointerVec& getPreOperand() const {
            return m_pre;
        }
        PointerVec& getPostOperand() {
            return m_post;
        }
        const PointerVec& getPostOperand() const {
            return m_post;
        }

        virtual void setFunction(Function f){
            m_f = std::move(f);
        }

        virtual void setDFunction(Function df){
            m_df = std::move(df);
        }

        void resetDFunction() {
            m_df = [this](const PointerVec&){
                auto val = eval();
                return T::Ones(val.rows(), val.cols());
            };
        }

        virtual void reset_value(){
            if(m_value.has_value()){
                m_value.reset();
                if(m_post.empty()){
                    reset_delta();
                } else {
                    for (auto &p: m_post) p->reset_value();
                }
            }
        }

        virtual void reset_delta(){
            if(m_delta.has_value()){
                m_delta.reset();
                for(auto& p: m_pre) p->reset_delta();
            }
        }

        bool isOptimizable() const {
            return m_optimizable;
        }

    protected:
        PointerVec m_pre, m_post;
        Function m_f, m_df;
        std::optional<T> m_value, m_delta;

        const bool m_optimizable;

        explicit Operand(bool optimizable)
                : m_pre(), m_post(),
                  m_f(std::nullopt), m_df(std::nullopt),
                  m_value(std::nullopt), m_delta(std::nullopt),
                  m_optimizable(optimizable){
            resetDFunction();
        }
    };

    template<typename T, typename ...Types>
    decltype(auto) make_operand(Types ...args){
        return std::make_shared<Operand<T>>(args...);
    }
}

#endif //LAZYDEEP1_OPERAND_HPP
