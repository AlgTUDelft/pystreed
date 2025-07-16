#include "solver/cache.h"

namespace STreeD {

	template <class OT>
	bool Cache<OT>::IsOptimalAssignmentCached(ADataView& data, const Branch& branch, int depth, int num_nodes) {
		if (!use_optimal_caching) return false;

		if (use_branch_caching && branch_cache.IsOptimalAssignmentCached(data, branch, depth, num_nodes)) return true;

		if (use_dataset_caching && dataset_cache.IsOptimalAssignmentCached(data, branch, depth, num_nodes)) return true;

		return false;
	}

	template <class OT>
	void Cache<OT>::StoreOptimalBranchAssignment(ADataView& data, const Branch& branch, SolContainer& opt_sols, int depth, int num_nodes) {
		if (!use_optimal_caching) return;
		if (use_branch_caching) branch_cache.StoreOptimalBranchAssignment(data, branch, opt_sols, depth, num_nodes);
		if (use_dataset_caching) dataset_cache.StoreOptimalBranchAssignment(data, branch, opt_sols, depth, num_nodes);
	}

	template <class OT>
	typename Cache<OT>::SolContainer Cache<OT>::RetrieveOptimalAssignment(ADataView& data, const Branch& branch, int depth, int num_nodes) {
		if (!use_optimal_caching) return empty_sol;
		if (use_branch_caching) {
			auto result = branch_cache.RetrieveOptimalAssignment(data, branch, depth, num_nodes);
			if (!CheckEmptySol<OT>(result)) return result;
		}
		if (use_dataset_caching) {
			auto result = dataset_cache.RetrieveOptimalAssignment(data, branch, depth, num_nodes);
			if (!CheckEmptySol<OT>(result)) return result;
		}
		return empty_sol;
	}

	template <class OT>
	void Cache<OT>::UpdateLowerBound(ADataView& data, const Branch& branch, typename Cache<OT>::SolContainer& lower_bound, int depth, int num_nodes) {
		runtime_assert(depth <= num_nodes);
		if (!use_lower_bound_caching) { return; }
		
		SolClearTemp<OT>(lower_bound);
		
		if (use_branch_caching) branch_cache.UpdateLowerBound(data, branch, lower_bound, depth, num_nodes);
		if (use_dataset_caching) dataset_cache.UpdateLowerBound(data, branch, lower_bound, depth, num_nodes);
	}
	
	template <class OT>
	typename Cache<OT>::SolContainer Cache<OT>::RetrieveLowerBound(ADataView& data, const Branch& branch, int depth, int num_nodes) {
		if (!use_lower_bound_caching) return empty_lb;
		if (use_branch_caching) {
			auto result = branch_cache.RetrieveLowerBound(data, branch, depth, num_nodes);
			if (!CheckEmptySol<OT>(result)) return result;
		}
		if (use_dataset_caching) {
			auto result = dataset_cache.RetrieveLowerBound(data, branch, depth, num_nodes);
			if (!CheckEmptySol<OT>(result)) return result;
		}
		return empty_lb;
	}

	template <class OT>
	void Cache<OT>::TransferAssignmentsForEquivalentBranches(const ADataView& data_source, const Branch& branch_source, const ADataView& data_destination, const Branch& branch_destination) {
		if (!use_lower_bound_caching) { return; }
		if (branch_source == branch_destination) { return; }

		if (use_branch_caching) branch_cache.TransferAssignmentsForEquivalentBranches(data_source, branch_source, data_destination, branch_destination);
		// Dataset caching do not need to transfer equivalent branches
	}

	template <class OT>
	void Cache<OT>::UpdateMaxDepthSearched(ADataView& data, const Branch& branch, int depth) {
		if (use_branch_caching) branch_cache.UpdateMaxDepthSearched(data, branch, depth);
		if (use_dataset_caching) dataset_cache.UpdateMaxDepthSearched(data, branch, depth);
	}

	template <class OT>
	int Cache<OT>::GetMaxDepthSearched(ADataView& data, const Branch& branch) {
		return std::max(
			use_branch_caching ? branch_cache.GetMaxDepthSearched(data, branch) : 0,
			use_dataset_caching ? dataset_cache.GetMaxDepthSearched(data, branch) : 0
		);
	}

	template class Cache<Accuracy>;
	template class Cache<CostComplexAccuracy>;
	template class Cache<BalancedAccuracy>;

	template class Cache<Regression>;
	template class Cache<CostComplexRegression>;
	template class Cache<PieceWiseLinearRegression>;
	template class Cache<SimpleLinearRegression>;

	template class Cache<CostSensitive>;
	template class Cache<InstanceCostSensitive>;
	template class Cache<F1Score>;
	template class Cache<GroupFairness>;
	template class Cache<EqOpp>;
	template class Cache<PrescriptivePolicy>;
	template class Cache<SurvivalAnalysis>;

}