/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#pragma once
#include "solver/cache_entry.h"

namespace STreeD {

	//key: a dataset
	//value: cached value contains the optimal value and the lower bound
	template <class OT>
	class DatasetCache  {
	public:

		using SolType = typename OT::SolType;
		using SolContainer = typename std::conditional<OT::total_order, Node<OT>, std::shared_ptr<Container<OT>>>::type;

		DatasetCache() = delete;
		DatasetCache(int max_branch_length);

		//related to storing/retriving optimal branches
		bool IsOptimalAssignmentCached(ADataView&, const Branch& branch, int depth, int num_nodes);
		void StoreOptimalBranchAssignment(ADataView&, const Branch& branch, SolContainer opt_sols, int depth, int num_nodes);
		SolContainer RetrieveOptimalAssignment(ADataView&, const Branch& branch, int depth, int num_nodes);
		//void TransferAssignmentsForEquivalentBranches(const ADataView&, const Branch& branch_source, const ADataView&, const Branch& branch_destination);//this updates branch_destination with all solutions from branch_source. Should only be done if the branches are equivalent.

		//related to storing/retrieving lower bounds
		void UpdateLowerBound(ADataView&, const Branch& branch, const SolContainer& lower_bound, int depth, int num_nodes);
		SolContainer RetrieveLowerBound(ADataView&, const Branch& branch, int depth, int num_nodes);

		int GetMaxDepthSearched(ADataView& data, const Branch& branch);
		void UpdateMaxDepthSearched(ADataView& data, const Branch& branch, int depth);

		int NumEntries() const;



		//we store a few iterators that were previous used in case they will be used in the future
		//useful when we query the cache multiple times for the exact same query
		struct PairIteratorBranch {
			typename std::unordered_map<ADataViewBitSet, CacheEntryVector<OT>>::iterator iter;
			Branch branch;
		};

		void InvalidateStoredIterators(ADataViewBitSet& data);
		
	private:

		//cache[i] is a hash table with datasets of size i	
		std::vector<std::unordered_map<ADataViewBitSet, CacheEntryVector<OT>>> cache;

		typename std::unordered_map<ADataViewBitSet, CacheEntryVector<OT>>::iterator FindIterator(ADataViewBitSet& data, const Branch& branch);
		std::vector<std::deque<PairIteratorBranch> > stored_iterators;

		SolContainer empty_sol;
	};

}