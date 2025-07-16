#pragma once
#include "base.h"
#include "solver/optimization_utils.h"
#include "tasks/tasks.h"

namespace STreeD {

	template <class OT>
	struct CacheEntry {

		using SolType = typename OT::SolType;
		using SolContainer = typename std::conditional<OT::total_order, Node<OT>, std::shared_ptr<Container<OT>>>::type;

		CacheEntry(int depth, int num_nodes) :
			depth(depth),
			num_nodes(num_nodes) {
			runtime_assert(depth <= num_nodes);
			lower_bound = InitializeSol<OT>(true);

		}

		CacheEntry(int depth, int num_nodes, const SolContainer& solutions) :
			optimal_solutions(solutions),
			lower_bound(solutions),
			depth(depth),
			num_nodes(num_nodes) {
			runtime_assert(depth <= num_nodes);
			runtime_assert(!CheckEmptySol<OT>(solutions));
		}

		SolContainer GetOptimalSolution() const {
			runtime_assert(IsOptimal());
			return CopySol<OT>(optimal_solutions);
		}


		inline const SolContainer& GetLowerBound() const { return lower_bound; }

		void SetOptimalSolutions(const SolContainer& optimal_solutions) {
			runtime_assert(!IsOptimal());
			runtime_assert(!CheckEmptySol<OT>(optimal_solutions));
			this->optimal_solutions = optimal_solutions;
			if (!CheckEmptySol<OT>(this->optimal_solutions)) {
				lower_bound = optimal_solutions;
			}
		}

		void UpdateLowerBound(const SolContainer& lower_bound) {
			runtime_assert(!IsOptimal());
			AddSolsInv<OT>(this->lower_bound, lower_bound);
		}

		inline bool IsOptimal() const { return !CheckEmptySol<OT>(optimal_solutions); }

		inline int GetNodeBudget() const { return num_nodes; }

		inline int GetDepthBudget() const { return depth; }

	private:
		SolContainer optimal_solutions;
		SolContainer lower_bound;
		int depth;
		int num_nodes;
	};

	template <class OT>
	struct CacheEntryVector {
		using SolType = typename OT::SolType;
		using SolContainer = typename std::conditional<OT::total_order, Node<OT>, std::shared_ptr<Container<OT>>>::type;

		CacheEntryVector() = default;
		CacheEntryVector(int size, const CacheEntry<OT>& _default) : entries(size, _default) {}

		void push_back(const CacheEntry<OT>& entry) { entries.push_back(entry); }
		CacheEntry<OT>& operator[](size_t idx) { return entries[idx]; }

		void UpdateMaxDepthSearched(int max_depth) { max_depth_searched = std::max(max_depth_searched, max_depth); }
		int GetMaxDepthSearched() const { return max_depth_searched; }

		int max_depth_searched{ 0 };
		std::vector<CacheEntry<OT>> entries;
	};
}