/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#include "solver/dataset_cache.h"

namespace STreeD {

	template <class OT>
	DatasetCache<OT>::DatasetCache(int num_instances) :
		cache(num_instances + 1),
		stored_iterators(num_instances + 1) {
		empty_sol = InitializeSol<OT>();
	}

	template <class OT>
	bool DatasetCache<OT>::IsOptimalAssignmentCached(ADataView& data, const Branch& branch, int depth, int num_nodes) {
		runtime_assert(depth <= num_nodes);

		auto& hashmap = cache[data.Size()];
		auto& bitsetview = data.GetBitSetView();
		auto iter = FindIterator(bitsetview, branch);

		if (iter == hashmap.end()) { return false; }

		for (CacheEntry<OT>& entry : iter->second) {
			if (entry.GetNodeBudget() == num_nodes && entry.GetDepthBudget() == depth) { 
				return entry.IsOptimal();
			}
		}
		return false;
	}

	template <class OT>
	void DatasetCache<OT>::StoreOptimalBranchAssignment(ADataView& data, const Branch& b, DatasetCache<OT>::SolContainer optimal_solutions, int depth, int num_nodes) {
		runtime_assert(depth <= num_nodes && num_nodes > 0);

		auto& hashmap = cache[data.Size()];
		auto& bitsetview = data.GetBitSetView();
		auto iter_vector_entry = FindIterator(bitsetview, b);// hashmap.find(data);
		int sol_num_nodes = num_nodes;
		if constexpr (OT::total_order) {
			sol_num_nodes = optimal_solutions.NumNodes();
		}
		int optimal_node_depth = std::min(depth, sol_num_nodes); //this is an estimate of the depth, it could be lower actually. We do not consider lower for simplicity, but it would be good to consider it as well.

		//if the branch has never been seen before, create a new entry for it
		if (iter_vector_entry == hashmap.end()) {
			std::vector<CacheEntry<OT>> vector_entry;
			for (int node_budget = sol_num_nodes; node_budget <= num_nodes; node_budget++) {
				for (int depth_budget = optimal_node_depth; depth_budget <= std::min(depth, node_budget); depth_budget++) {
					vector_entry.push_back({ depth_budget, node_budget, optimal_solutions });
				}
			}
			if (!data.IsHashSet()) { data.SetHash(std::hash<ADataViewBitSet>()(bitsetview)); }
			cache[data.Size()].insert(std::pair<ADataViewBitSet, std::vector<CacheEntry<OT>> >(bitsetview, vector_entry));
			InvalidateStoredIterators(bitsetview);
		} else {
			//this sol is valid for size=[opt.NumNodes, num_nodes] and depths d=min(size, depth)

			//now we need to see if other node budgets have been seen before. 
			//For each budget that has been seen, update it;
			std::vector<std::vector<bool> > budget_seen(size_t(num_nodes) + 1, std::vector<bool>(depth + 1, false));
			for (CacheEntry<OT>& entry : iter_vector_entry->second) {
				//todo enable this here! //runtime_assert(optimal_node.Misclassifications() >= entry.GetLowerBound() || optimal_node.NumNodes() > entry.GetNodeBudget());

				//I believe it rarely happens that we receive a solution with less nodes than 'num_nodes', but it is possible
				if (sol_num_nodes <= entry.GetNodeBudget() && entry.GetNodeBudget() <= num_nodes
					&& optimal_node_depth <= entry.GetDepthBudget() && entry.GetDepthBudget() <= depth) {
					/*if (!(!entry.IsOptimal() || entry.GetOptimalValue() == optimal_solutions->Misclassifications())) {
						std::cout << "opt node: " << optimal_solutions->NumNodes() << ", " << optimal_node.misclassification_score << "\n";
						std::cout << "\tnum nodes: " << num_nodes << "\n";
						std::cout << entry.GetNodeBudget() << ", " << entry.GetOptimalValue() << "\n";
					}*/ // TODO fix this
					// runtime_assert(!entry.IsOptimal() || entry.GetOptimalValue() == optimal_node.Misclassifications()); TODO enable this

					{
						budget_seen[entry.GetNodeBudget()][entry.GetDepthBudget()] = true;
						if (!entry.IsOptimal()) {
							entry.SetOptimalSolutions(optimal_solutions);
						}
					}

					runtime_assert(entry.GetDepthBudget() <= entry.GetNodeBudget()); //fix the case when it turns out that more nodes do not give a better result...e.g., depth 4 and num nodes 4, but a solution with three nodes found...that solution is then optimal for depth 3 as well...need to update but lazy now
				}
			}
			//create entries for those which were not seen
			//note that most of the time this loop only does one iteration since usually using the full node budget gives better results
			for (int node_budget = sol_num_nodes; node_budget <= num_nodes; node_budget++) {
				for (int depth_budget = optimal_node_depth; depth_budget <= std::min(depth, node_budget); depth_budget++) {
					if (!budget_seen[node_budget][depth_budget]) {
						CacheEntry<OT> entry(depth_budget, node_budget, optimal_solutions);
						iter_vector_entry->second.push_back(entry);
						runtime_assert(entry.GetDepthBudget() <= entry.GetNodeBudget()); //todo no need for this assert
					}
				}
			}
		}
		//TODO: the cache needs to invalidate out solutions that are dominated, i.e., with the same objective value but less nodes
		//or I need to rethink this caching to include exactly num_nodes -> it might be strange that we ask for five nodes and get UNSAT, while with four nodes it gives a solution
		//I am guessing that the cache must store exactly num_nodes, and then outside the loop when we find that the best sol has less node, we need to insert that in the cache?
		//and mark all solutions with more nodes as infeasible, i.e., some high cost
		
		// TODO check solution quality of cache with new found solution
		//auto opt = RetrieveOptimalAssignment(data, b, depth, num_nodes);
		//runtime_assert(opt == optimal_solutions);
	}

	/*template <class OT>
	void DatasetCache::TransferAssignmentsForEquivalentBranches(const ADataView&, const Branch& branch_source, const ADataView&, const Branch& branch_destination) {
		return; //no need to transfer when caching datasets
	}*/

	template <class OT>
	typename DatasetCache<OT>::SolContainer DatasetCache<OT>::RetrieveOptimalAssignment(ADataView& data, const Branch& b, int depth, int num_nodes) {
		auto& hashmap = cache[data.Size()];
		auto& bitsetview = data.GetBitSetView();
		auto iter = FindIterator(bitsetview, b);// hashmap.find(data);

		if (iter == hashmap.end()) { return empty_sol; }

		for (CacheEntry<OT>& entry : iter->second) {
			if (entry.GetDepthBudget() == depth && entry.GetNodeBudget() == num_nodes && entry.IsOptimal()) {
				return entry.GetOptimalSolution();
			}
		}
		return empty_sol;
	}

	template <class OT>
	void DatasetCache<OT>::UpdateLowerBound(ADataView& data, const Branch& branch, const typename DatasetCache<OT>::SolContainer& lower_bound, int depth, int num_nodes) {
		runtime_assert(depth <= num_nodes);

		auto& hashmap = cache[data.Size()];
		auto& bitsetview = data.GetBitSetView();
		auto iter_vector_entry = FindIterator(bitsetview, branch);// hashmap.find(data);

		//if the branch has never been seen before, create a new entry for it
		if (iter_vector_entry == hashmap.end()) {
			std::vector<CacheEntry<OT>> vector_entry(1, CacheEntry<OT>(depth, num_nodes)); 
			vector_entry[0].UpdateLowerBound(lower_bound);
			if (!data.IsHashSet()) { data.SetHash(std::hash<ADataViewBitSet>()(bitsetview)); }
			cache[data.Size()].insert(std::pair<ADataViewBitSet, std::vector<CacheEntry<OT>> >(bitsetview, vector_entry));
			InvalidateStoredIterators(bitsetview);
		} else {
			//now we need to see if this node node_budget has been seen before. 
			//If it was seen, update it; otherwise create a new entry
			bool found_corresponding_entry = false;
			for (CacheEntry<OT>& entry : iter_vector_entry->second) {
				if (entry.GetDepthBudget() == depth && entry.GetNodeBudget() == num_nodes) {
					entry.UpdateLowerBound(lower_bound);
					found_corresponding_entry = true;
					break;
				}
			}

			if (!found_corresponding_entry) {
				CacheEntry<OT> entry(depth, num_nodes);
				entry.UpdateLowerBound(lower_bound);
				iter_vector_entry->second.push_back(entry);
			}
		}
	}

	template <class OT>
	typename DatasetCache<OT>::SolContainer DatasetCache<OT>::RetrieveLowerBound(ADataView& data, const Branch& b, int depth, int num_nodes) {
		runtime_assert(depth <= num_nodes);

		auto& hashmap = cache[data.Size()];
		auto& bitsetview = data.GetBitSetView();
		auto iter = FindIterator(bitsetview, b);// hashmap.find(data);

		auto best_lower_bound = InitializeSol<OT>(true);
		if (iter == hashmap.end()) { return best_lower_bound; }

		//compute the misclassification lower bound by considering that branches with more node/depth budgets 
		//  can only have less or equal misclassification than when using the prescribed number of nodes and depth
		for (CacheEntry<OT>& entry : iter->second) {
			if (num_nodes <= entry.GetNodeBudget() && depth <= entry.GetDepthBudget()) { 
				auto& local_lower_bound = entry.GetLowerBound();
				if (!CheckEmptySol<OT>(local_lower_bound)) {
					if (CheckEmptySol<OT>(best_lower_bound)) {
						best_lower_bound = CopySol<OT>(local_lower_bound);
					} else {
						AddSolsInv<OT>(best_lower_bound, local_lower_bound);
					}
				}
			}
		}
		return best_lower_bound;
	}

	template <class OT>
	int DatasetCache<OT>::NumEntries() const {
		size_t count = 0;
		for (auto& c : cache) {
			count += c.size();
		}
		return int(count);
	}

	template <class OT>
	void DatasetCache<OT>::InvalidateStoredIterators(ADataViewBitSet& data) {
		stored_iterators[data.Size()].clear();
	}

	template <class OT>
	typename std::unordered_map<ADataViewBitSet, std::vector<CacheEntry<OT>>>::iterator DatasetCache<OT>::FindIterator(ADataViewBitSet& data, const Branch& branch) {
		for (PairIteratorBranch& p : stored_iterators[data.Size()]) {
			if (p.branch == branch) { return p.iter; }
		}

		if (!data.IsHashSet()) { data.SetHash(std::hash<ADataViewBitSet>()(data)); }

		auto& hashmap = cache[data.Size()];
		auto iter = hashmap.find(data);

		PairIteratorBranch hehe;
		hehe.branch = branch;
		hehe.iter = iter;

		if (stored_iterators[data.Size()].size() == 2) { stored_iterators[data.Size()].pop_back(); }
		stored_iterators[data.Size()].push_front(hehe);
		return iter;
	}

	template class DatasetCache<Accuracy>;
	template class DatasetCache<CostComplexAccuracy>;

	template class DatasetCache<Regression>;
	template class DatasetCache<CostComplexRegression>;
	template class DatasetCache<PieceWiseLinearRegression>;
	template class DatasetCache<SimpleLinearRegression>;

	template class DatasetCache<CostSensitive>;
	template class DatasetCache<InstanceCostSensitive>;
	template class DatasetCache<F1Score>;
	template class DatasetCache<GroupFairness>;
	template class DatasetCache<EqOpp>;
	template class DatasetCache<PrescriptivePolicy>;
	template class DatasetCache<SurvivalAnalysis>;

}