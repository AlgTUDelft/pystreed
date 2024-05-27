/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#include "solver/branch_cache.h"

namespace STreeD {

	template <class OT>
	bool BranchCache<OT>::IsOptimalAssignmentCached(ADataView& data, const Branch& branch, int depth, int num_nodes) {
		runtime_assert(depth <= num_nodes);

		auto& hashmap = cache[branch.Depth()];
		auto iter = hashmap.find(branch);

		if (iter == hashmap.end()) { return false; }

		for (const CacheEntry<OT>& entry : iter->second) {
			if (entry.GetNodeBudget() == num_nodes && entry.GetDepthBudget() == depth) {
				return entry.IsOptimal();
			}
		}
		return false;
	}

	template <class OT>
	void BranchCache<OT>::StoreOptimalBranchAssignment(ADataView& data, const Branch& branch, BranchCache<OT>::SolContainer& optimal_solutions, int depth, int num_nodes) {
		runtime_assert(depth <= num_nodes && num_nodes > 0);

		SolClearTemp<OT>(optimal_solutions);

		auto& hashmap = cache[branch.Depth()];
		auto iter_vector_entry = hashmap.find(branch);
		int sol_num_nodes = num_nodes;
		if constexpr (OT::total_order) {
			sol_num_nodes = optimal_solutions.NumNodes();
		}
		int optimal_node_depth = std::min(depth, num_nodes); //this is an estimate of the depth, it could be lower actually. We do not consider lower for simplicity, but it would be good to consider it as well.

		//if the branch has never been seen before, create a new entry for it
		if (iter_vector_entry == hashmap.end()) {
			std::vector<CacheEntry<OT>> vector_entry;
			for (int node_budget = sol_num_nodes; node_budget <= num_nodes; node_budget++) {
				for (int depth_budget = optimal_node_depth; depth_budget <= std::min(depth, node_budget); depth_budget++) {
					vector_entry.push_back({ depth_budget, node_budget, optimal_solutions });
				}
			}
			cache[branch.Depth()].insert(std::pair<Branch, std::vector<CacheEntry<OT>> >(branch, vector_entry));
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
					/*if (!(!entry.IsOptimal() || entry.GetOptimalValue() == optimal_node.Misclassifications())) {
						std::cout << "opt node: " << optimal_node.NumNodes() << ", " << optimal_node.misclassification_score << "\n";
						std::cout << "\tnum nodes: " << num_nodes << "\n";
						std::cout << entry.GetNodeBudget() << ", " << entry.GetOptimalValue() << "\n";
					} // TODO fix this
					runtime_assert(!entry.IsOptimal() || entry.GetOptimalValue() == optimal_node.Misclassifications());*/

					budget_seen[entry.GetNodeBudget()][entry.GetDepthBudget()] = true;
					if (!entry.IsOptimal()) { entry.SetOptimalSolutions(optimal_solutions); }
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
		//TODO fix this statement
		//runtime_assert(RetrieveOptimalAssignment(data, branch, depth, num_nodes).Misclassifications() == optimal_node.Misclassifications());
	}

	template <class OT>
	void BranchCache<OT>::TransferAssignmentsForEquivalentBranches(const ADataView&, const Branch& branch_source, const ADataView&, const Branch& branch_destination) {
		
		auto& hashmap = cache[branch_source.Depth()];
		auto iter_source = hashmap.find(branch_source);
		auto iter_destination = hashmap.find(branch_destination);

		if (iter_source == hashmap.end()) { return; }

		if (iter_destination == hashmap.end()) //if the branch has never been seen before, create a new entry for it and copy everything into it
		{
			std::vector<CacheEntry<OT>> vector_entry = iter_source->second;
			cache[branch_destination.Depth()].insert(std::pair<Branch, std::vector<CacheEntry<OT>> >(branch_destination, vector_entry));
		} else {
			for (CacheEntry<OT>& entry_source : iter_source->second) {
				//todo could be done more efficiently
				bool should_add = true;
				for (CacheEntry<OT>& entry_destination : iter_destination->second) {
					if (entry_source.GetDepthBudget() == entry_destination.GetDepthBudget() &&
						entry_source.GetNodeBudget() == entry_destination.GetNodeBudget()) {
						should_add = false;
						//if the source entry is strictly better than the destination entry, replace it
						if (entry_source.IsOptimal() && !entry_destination.IsOptimal() 
							|| LeftStrictDominatesRight<OT>(entry_source.GetLowerBound(), entry_destination.GetLowerBound())) {
							entry_destination = entry_source;
							break;
						}
					}
				}
				if (should_add) { iter_destination->second.push_back(entry_source); }
			}
		}
	}

	template <class OT>
	typename BranchCache<OT>::SolContainer BranchCache<OT>::RetrieveOptimalAssignment(ADataView& data, const Branch& branch, int depth, int num_nodes) {
		auto& hashmap = cache[branch.Depth()];
				
		auto iter = hashmap.find(branch);
		if (iter == hashmap.end()) { return empty_sol; }

		for (CacheEntry<OT>& entry : iter->second) {
			if (entry.GetDepthBudget() == depth && entry.GetNodeBudget() == num_nodes && entry.IsOptimal()) {
				return entry.GetOptimalSolution();
			}
		}
		return empty_sol;
	}

	template <class OT>
	void BranchCache<OT>::UpdateLowerBound(ADataView& data, const Branch& branch, const typename BranchCache<OT>::SolContainer& lower_bound, int depth, int num_nodes) {
		runtime_assert(depth <= num_nodes);

		auto& hashmap = cache[branch.Depth()];
		auto iter_vector_entry = hashmap.find(branch);

		//if the branch has never been seen before, create a new entry for it
		if (iter_vector_entry == hashmap.end()) {
			std::vector<CacheEntry<OT>> vector_entry(1, CacheEntry<OT>(depth, num_nodes));
			vector_entry[0].UpdateLowerBound(lower_bound);
			cache[branch.Depth()].insert(std::pair<Branch, std::vector<CacheEntry<OT>> >(branch, vector_entry));
		} else {
			//now we need to see if this node node_budget has been seen before. 
			//If it was seen, update it; otherwise create a new entry
			bool found_corresponding_entry = false;
			for (CacheEntry<OT>& entry : iter_vector_entry->second) {
				//If the new lower bound is found with more relaxed discrimination budget, 
				// yet it has a higher lower-bound, update this entry
				if (entry.GetDepthBudget() == depth && entry.GetNodeBudget() == num_nodes) {
					if(!entry.IsOptimal())
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
	typename BranchCache<OT>::SolContainer BranchCache<OT>::RetrieveLowerBound(ADataView& data, const Branch& branch, int depth, int num_nodes) {
		runtime_assert(depth <= num_nodes);
		
		auto& hashmap = cache[branch.Depth()];
		auto iter = hashmap.find(branch);

		if (iter == hashmap.end()) { return empty_lb; }

		//compute the misclassification lower bound by considering that branches with more node/depth budgets 
		//  can only have less or equal misclassification than when using the prescribed number of nodes and depth
		
		auto best_lower_bound = InitializeSol<OT>(true);
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
	int BranchCache<OT>::NumEntries() const {
		size_t count = 0;
		for (auto& c : cache) {
			count += c.size();
		}
		return int(count);
	}

	template class BranchCache<Accuracy>;
	template class BranchCache<CostComplexAccuracy>;

	template class BranchCache<Regression>;
	template class BranchCache<CostComplexRegression>;
	template class BranchCache<PieceWiseLinearRegression>;
	template class BranchCache<SimpleLinearRegression>;

	template class BranchCache<CostSensitive>;
	template class BranchCache<InstanceCostSensitive>;
	template class BranchCache<F1Score>;
	template class BranchCache<GroupFairness>;
	template class BranchCache<EqOpp>;
	template class BranchCache<PrescriptivePolicy>;
	template class BranchCache<SurvivalAnalysis>;

}