#pragma once

#include <stdint.h>

#include "semantic_program.cuh"

namespace semantic_rule_action {

enum class PrefixWorkEvent { TableSlot, Category };
// Standalone structural callers do not own an execution accumulator. Resident
// production prefix callers always supply the original thread-local charge.
struct UnaccountedPrefixWork {
    __device__ void operator()(PrefixWorkEvent) const {}
};

// NULL and intrinsic fields retain their meaning. Context-specific rows are
// namespaced by original occurrence; external support cannot grant foreign rows.
template <typename Charge = UnaccountedPrefixWork>
__device__ bool context_category(const uint64_t* books, uint32_t field,
                                uint32_t category, Charge charge = {}) {
    if (!category) return true;
    const uint32_t range = field == 2 ? 0 : field >= 3 && field <= 10 ? 1 :
                           field == 11 ? 2 : field == 17 ? 3 : UINT32_MAX;
    if (range == UINT32_MAX) return true;
    charge(PrefixWorkEvent::TableSlot);
    const uint64_t begin = books[29 + 2 * range];
    charge(PrefixWorkEvent::TableSlot);
    const uint64_t end = books[30 + 2 * range];
    return category >= begin && category < end;
}

// Structural descriptors occupy disjoint, typed ranges after the ordinary
// graph descriptors. No ordinary value or statement target decodes as a rule.
template <typename Charge = UnaccountedPrefixWork>
__device__ bool head_predicate(const uint64_t* books, uint32_t category,
                               uint32_t predicate_count, uint32_t* predicate,
                               Charge charge = {}) {
    charge(PrefixWorkEvent::Category);
    if (!books || !predicate || !category || category >= books[1] ||
        !context_category(books, 2, category, charge)) return false;
    const uint64_t* row = books + books[0] + 10 * uint64_t(category);
    if ((charge(PrefixWorkEvent::TableSlot), row[1] != UINT64_MAX) ||
        (charge(PrefixWorkEvent::TableSlot), row[0] >= predicate_count)) return false;
    for (uint32_t index = 2; index < 10; ++index) {
        charge(PrefixWorkEvent::TableSlot);
        if (row[index]) return false;
    }
    *predicate = uint32_t(row[0]);
    return true;
}

template <typename Charge = UnaccountedPrefixWork>
__device__ bool operand(const uint64_t* books, uint32_t category,
                        uint32_t kind, uint32_t bound, uint32_t* value,
                        Charge charge = {}) {
    charge(PrefixWorkEvent::Category);
    if (!books || !value || !category || category >= books[3] ||
        !context_category(books, 3, category, charge)) return false;
    const uint64_t* row = books + books[2] + 6 * uint64_t(category);
    if ((charge(PrefixWorkEvent::TableSlot), row[0]) ||
        (charge(PrefixWorkEvent::TableSlot), row[1]) ||
        (charge(PrefixWorkEvent::TableSlot), row[2]) ||
        (charge(PrefixWorkEvent::TableSlot), row[3] != kind) ||
        (charge(PrefixWorkEvent::TableSlot), row[4] >= bound) ||
        (charge(PrefixWorkEvent::TableSlot), row[5])) return false;
    *value = uint32_t(row[4]);
    return true;
}

template <typename Charge = UnaccountedPrefixWork>
__device__ bool body_predicate(const uint64_t* books, uint32_t category,
                               uint32_t predicate_count, uint32_t* predicate,
                               Charge charge = {}) {
    return operand(books, category, 1, predicate_count, predicate, charge);
}

template <typename Charge = UnaccountedPrefixWork>
__device__ bool variable(const uint64_t* books, uint32_t category,
                         uint32_t* index, Charge charge = {}) {
    return operand(books, category, 2, 4, index, charge);
}

template <typename Charge = UnaccountedPrefixWork>
__device__ bool decode(const uint64_t* books, const uint32_t choices[18],
                       uint32_t predicate_count, semantic_program::Rule* rule,
                       Charge charge = {}) {
    charge(PrefixWorkEvent::Category);
    if (!books || !choices || !rule || !predicate_count || choices[0] != 1 ||
        choices[1] != 2) return false;
    for (uint32_t field = 11; field < 18; ++field) {
        charge(PrefixWorkEvent::TableSlot);
        charge(PrefixWorkEvent::Category);
        if (choices[field]) return false;
    }
    semantic_program::Rule decoded{};
    decoded.body_count = 2;
    if (!head_predicate(books, choices[2], predicate_count,
                        &decoded.head_predicate, charge) ||
        !body_predicate(books, choices[3], predicate_count,
                        &decoded.left_predicate, charge) ||
        !body_predicate(books, choices[4], predicate_count,
                        &decoded.right_predicate, charge) ||
        !variable(books, choices[5], &decoded.head_variables[0], charge) ||
        !variable(books, choices[6], &decoded.head_variables[1], charge) ||
        !variable(books, choices[7], &decoded.left_variables[0], charge) ||
        !variable(books, choices[8], &decoded.left_variables[1], charge) ||
        !variable(books, choices[9], &decoded.right_variables[0], charge) ||
        !variable(books, choices[10], &decoded.right_variables[1], charge) ||
        !(charge(PrefixWorkEvent::Category), semantic_program::valid_rule(decoded, predicate_count))) return false;
    charge(PrefixWorkEvent::TableSlot);
    *rule = decoded;
    return true;
}

template <typename Component, typename Charge = UnaccountedPrefixWork>
__device__ bool prefix_completes(const uint64_t* books,
                                 const Component* components,
                                 const uint8_t* support, uint32_t start,
                                 const uint32_t choices[18], uint32_t field,
                                 uint32_t category, uint32_t predicate_count,
                                 Charge charge = {}) {
    charge(PrefixWorkEvent::Category);
    if (!books || !components || !support || !choices || !predicate_count ||
        field < 1 || field >= 18 || books[1] <= uint64_t(predicate_count) ||
        books[3] <= uint64_t(predicate_count) + 4 ||
        (field == 1 ? category != 2 : choices[1] != 2)) return false;
    charge(PrefixWorkEvent::TableSlot);
    const uint32_t head_end = uint32_t(books[30]);
    charge(PrefixWorkEvent::TableSlot);
    const uint32_t body_end = uint32_t(books[32]);
    if (head_end < predicate_count || body_end < predicate_count + 4) return false;
    const uint32_t head_start = head_end - predicate_count;
    const uint32_t body_start = body_end - predicate_count - 4;
    const uint32_t variable_start = body_end - 4;
    auto admitted = [&](uint32_t position, uint32_t candidate) {
        charge(PrefixWorkEvent::Category);
        charge(PrefixWorkEvent::TableSlot);
        const Component component = components[start + position];
        return candidate < component.cardinality &&
               context_category(books, position, candidate, charge) &&
               support[component.offset + candidate] &&
               (position > field ||
                candidate == (position == field ? category : choices[position]));
    };
    for (uint32_t position = 11; position < 18; ++position)
        if (!admitted(position, 0)) return false;
    for (uint32_t position = 2; position <= 4; ++position) {
        charge(PrefixWorkEvent::TableSlot);
        if (position <= field) {
            const uint32_t fixed = position == field ? category : choices[position];
            const uint32_t begin = position == 2 ? head_start : body_start;
            uint32_t decoded = UINT32_MAX;
            charge(PrefixWorkEvent::Category);
            if (fixed < begin || fixed - begin >= predicate_count ||
                !admitted(position, fixed) ||
                !(position == 2 ? head_predicate(books, fixed, predicate_count, &decoded, charge)
                                 : body_predicate(books, fixed, predicate_count, &decoded, charge)) ||
                decoded != fixed - begin) return false;
            continue;
        }
        bool found = false;
        for (uint32_t predicate = 0; predicate < predicate_count; ++predicate) {
            charge(PrefixWorkEvent::TableSlot);
            const uint32_t candidate = (position == 2 ? head_start : body_start) + predicate;
            uint32_t decoded = UINT32_MAX;
            if (admitted(position, candidate) &&
                (position == 2 ? head_predicate(books, candidate, predicate_count, &decoded, charge)
                               : body_predicate(books, candidate, predicate_count, &decoded, charge)) &&
                decoded == predicate) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    auto available_variable = [&](uint32_t position, uint32_t index) {
        charge(PrefixWorkEvent::Category);
        const uint32_t candidate = variable_start + index;
        uint32_t decoded = UINT32_MAX;
        return admitted(position, candidate) && variable(books, candidate, &decoded, charge) &&
               decoded == index;
    };
    if (field >= 5 && field <= 10) {
        uint32_t decoded = UINT32_MAX;
        charge(PrefixWorkEvent::Category);
        if (category < variable_start || category - variable_start >= 4 ||
            !admitted(field, category) || !variable(books, category, &decoded, charge) ||
            decoded != category - variable_start) return false;
    }
    for (uint32_t first = 0; first < 4; ++first) {
        charge(PrefixWorkEvent::TableSlot);
        if (!available_variable(5, first)) continue;
        for (uint32_t second = 0; second < 4; ++second) {
            charge(PrefixWorkEvent::TableSlot);
            if (!available_variable(6, second)) continue;
            const uint32_t required = (1U << first) | (1U << second);
            uint32_t coverings = 1;
            for (uint32_t position = 7; position <= 10; ++position) {
                charge(PrefixWorkEvent::TableSlot);
                uint32_t next = 0;
                for (uint32_t covered = 0; covered < 16; ++covered) {
                    charge(PrefixWorkEvent::TableSlot);
                    charge(PrefixWorkEvent::Category);
                    if (!(coverings & (1U << covered))) continue;
                    for (uint32_t index = 0; index < 4; ++index)
                        if (available_variable(position, index))
                            next |= 1U << (covered | (1U << index));
                }
                coverings = next;
            }
            for (uint32_t covered = 0; covered < 16; ++covered) {
                charge(PrefixWorkEvent::TableSlot);
                charge(PrefixWorkEvent::Category);
                if ((coverings & (1U << covered)) && (covered & required) == required)
                    return true;
            }
        }
    }
    return false;
}

struct Evaluation {
    semantic_program::Result closure;
    uint64_t truth[3];
};

__device__ Evaluation evaluate(const semantic_program::Bank& bank,
                               const semantic_program::Rule* inserted,
                               uint32_t inserted_count,
                               semantic_program::Fact* input,
                               uint32_t input_capacity,
                               semantic_program::Rule* rules,
                               uint32_t rule_capacity,
                               semantic_program::Fact* output,
                               uint32_t output_capacity,
                               uint64_t fuel) {
    Evaluation evaluation{};
    evaluation.closure = semantic_program::materialize_bank(
        bank, inserted, inserted_count, input, input_capacity, rules,
        rule_capacity, output, output_capacity, fuel);
    if (evaluation.closure.status != semantic_program::Status::Ok) return evaluation;
    for (uint32_t query = 0; query < 3; ++query) {
        const auto fact = semantic_program::bank_fact(bank, query);
        evaluation.truth[query] = semantic_program::contains(
            output, evaluation.closure.fact_count, fact) ? 1 : 0;
    }
    return evaluation;
}

} // namespace semantic_rule_action
