/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#pragma once
#include "solver/cache_entry.h"

namespace STreeD {

	//key: a branch
	//value: cached value contains the optimal value and the lower bound
	template <class OT>
	class BranchCache {
	public:

		using SolType = typename OT::SolType;
		using SolContainer = typename std::conditional<OT::total_order, Node<OT>, std::shared_ptr<Container<OT>>>::type;

		BranchCache(int max_branch_length) : cache(max_branch_length) {
			empty_sol = InitializeSol<OT>();
			empty_lb = InitializeSol<OT>(true);
		}

		//related to storing/retriving optimal branches
		bool IsOptimalAssignmentCached(ADataView&, const Branch& branch, int depth, int num_nodes);
		void StoreOptimalBranchAssignment(ADataView&, const Branch& branch, SolContainer& optimal_solutions, int depth, int num_nodes);
		SolContainer RetrieveOptimalAssignment(ADataView&, const Branch& branch, int depth, int num_nodes);
		void TransferAssignmentsForEquivalentBranches(const ADataView&, const Branch& branch_source, const ADataView&, const Branch& branch_destination);//this updates branch_destination with all solutions from branch_source. Should only be done if the branches are equivalent.

		//related to storing/retrieving lower bounds
		void UpdateLowerBound(ADataView&, const Branch& branch, const SolContainer& lower_bound, int depth, int num_nodes);
		SolContainer RetrieveLowerBound(ADataView&, const Branch& branch, int depth, int num_nodes);

		int GetMaxDepthSearched(ADataView&, const Branch& branch);
		void UpdateMaxDepthSearched(ADataView& data, const Branch& branch, int depth);

		//misc
		int NumEntries() const;
		

	private:
		
		//cache[i] is a hash table with branches of size i		
		std::vector<std::unordered_map<Branch, CacheEntryVector<OT>, BranchHashFunction, BranchEquality >> cache; 

		SolContainer empty_sol, empty_lb;
	};
}