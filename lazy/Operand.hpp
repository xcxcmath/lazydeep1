//
// Created by bjk on 19. 1. 24.
//

#ifndef LAZYDEEP1_OPERAND_HPP
#define LAZYDEEP1_OPERAND_HPP

#include <Eigen>
#include <memory>
#include <algorithm>
#include <random>
#include <functional>
#include <optional>
#include <set>
#include <map>

#define LAZY_DELETED_FUNCTIONS(Base, T) \
    Base(const Base<T>&) = delete; \
    Base<T>& operator=(const Base<T>&) = delete; \
    Base(Base<T>&& rhs) = delete; \
    Base<T>& operator=(Base<T>&&) = delete;

namespace lazy {
    template<typename T>
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    using Index = Eigen::Index;

    template<typename T>
    class Operand {
    public:
        using ValueType = T;
        using Pointer = std::shared_ptr<Operand<T>>;
        using Function = std::function<T()>;
        using DFunction = std::function<T(const Pointer&)>;
        using PointerMap = std::map<Pointer, DFunction>;
        using PointerSet = std::set<Pointer>;

        /*
         * Constructors / Destructors / Assignment operators
         */

        explicit Operand()
        : m_f([](){return T();}), m_df(),
        m_pre(), m_post(),
        m_value(std::nullopt), m_delta(),
        m_optimizable(false) {

        }

        virtual ~Operand() = default;

        // Anything about Copy/Move is inhibited
        LAZY_DELETED_FUNCTIONS(Operand, T);

        /*
         * Evaluating methods
         */

        virtual const T& eval(){
            return m_value.has_value() ? m_value.value() : m_value.emplace(m_f());
        }

        virtual const T& diff(const Pointer& E){
            if(auto it = m_delta.find(E); it != m_delta.end()){
                return m_delta.at(E);
            }

            auto& cache = m_delta[E];
            const T& val = eval();
            cache = T::Zero(val.rows(), val.cols());

            if(m_post.empty()){
                if(E.get() == this){
                    cache = T::Ones(val.rows(), val.cols());
                }

                return cache;
            }

            for(const auto& [ptr, df]: m_df){
                cache = cache + df(E);
            }

            return cache;
        }

        /*
         * Getter/Setter
         */

        PointerSet& getPreOperand() noexcept {
            return m_pre;
        }
        const PointerSet& getPreOperand() const noexcept {
            return m_pre;
        }
        PointerSet& getPostOperand() noexcept {
            return m_post;
        }
        const PointerSet& getPostOperand() const noexcept {
            return m_post;
        }
        PointerMap& getDF() noexcept {
            return m_df;
        }
        const PointerMap& getDF() const noexcept {
            return m_df;
        }

        virtual void setFunction(Function f){
            m_f = std::move(f);
        }

        bool isOptimizable() const noexcept {
            return m_optimizable;
        }

        /*
         * Re-setter
         */

        virtual void resetValue(){
            if(m_value.has_value()){
                m_value.reset();
                if(m_post.empty()){
                    resetDelta();
                } else {
                    for (const auto &ptr: m_post) ptr->resetValue();
                }
            }
        }

        virtual void resetDelta(){
            if(!m_delta.empty()){
                m_delta.clear();
                for(const auto& p: m_pre) p->resetDelta();
            }
        }

    protected:
        Function m_f;
        PointerMap m_df;
        PointerSet m_pre;
        PointerSet m_post;

        std::optional<T> m_value;
        std::map<Pointer, T> m_delta;

        bool m_optimizable;
    };

    template<typename T, typename ...Types>
    decltype(auto) make_operand(Types ...args){
        return std::make_shared<Operand<T>>(args...);
    }
}

#endif //LAZYDEEP1_OPERAND_HPP
