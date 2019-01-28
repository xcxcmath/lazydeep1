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
#include <set>
#include <map>

namespace lazy {
    template<typename T>
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    template<typename T>
    class Operand {
    public:
        using ValueType = T;
        using Pointer = std::shared_ptr<Operand<T>>;
        using Function = std::function<T()>;
        using DFunction = std::function<T(const Pointer&)>;
        using PointerMap = std::map<Pointer, DFunction>;
        using PointerSet = std::set<Pointer>;

        explicit Operand()
        : m_f([](){return T();}),
        m_pre(), m_post(),
        m_value(std::nullopt), m_delta(std::nullopt),
        m_optimizable(false) {

        }

        virtual const T& eval(){
            return m_value.has_value() ? m_value.value() : m_value.emplace(m_f());
        }

        virtual const T& diff(const Pointer& E){
            if(m_delta.has_value())
                return m_delta.value();

            const T& val = eval();
            m_delta = T::Zero(val.rows(), val.cols());

            if(m_post.empty()){
                if(E.get() == this){
                    m_delta = T::Ones(val.rows(), val.cols());
                }

                return m_delta.value();
            }

            for(const auto& [ptr, df]: m_post){
                m_delta = m_delta.value() + df(E);
            }

            return m_delta.value();
        }

        PointerSet& getPreOperand() {
            return m_pre;
        }
        const PointerSet& getPreOperand() const {
            return m_pre;
        }
        PointerMap& getPostOperand() {
            return m_post;
        }
        const PointerMap& getPostOperand() const {
            return m_post;
        }

        virtual void setFunction(Function f){
            m_f = std::move(f);
        }

        virtual void reset_value(){
            if(m_value.has_value()){
                m_value.reset();
                if(m_post.empty()){
                    reset_delta();
                } else {
                    for (auto &[ptr, _]: m_post) ptr->reset_value();
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
        Function m_f;
        PointerSet m_pre;
        PointerMap m_post;

        std::optional<T> m_value, m_delta;

        const bool m_optimizable;

        explicit Operand(bool optimizable)
                : m_post(),
                  m_value(std::nullopt), m_delta(std::nullopt),
                  m_optimizable(optimizable){

        }
    };

    template<typename T, typename ...Types>
    decltype(auto) make_operand(Types ...args){
        return std::make_shared<Operand<T>>(args...);
    }
}

#endif //LAZYDEEP1_OPERAND_HPP
