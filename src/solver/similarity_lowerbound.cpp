/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/

#include "solver/similarity_lowerbound.h"

namespace STreeD {

	template <class OT>
	SimilarityLowerBoundComputer<OT>::SimilarityLowerBoundComputer(OT* optimization_task, int num_labels, int max_depth, int size, int num_instances) :
		disabled_(false) {
		Initialise(optimization_task, num_labels, max_depth, size);
	}

	template <class OT>
	typename SimilarityLowerBoundComputer<OT>::SolContainer SimilarityLowerBoundComputer<OT>::SubstractLB(
		typename SimilarityLowerBoundComputer<OT>::SolContainer& lb, typename SimilarityLowerBoundComputer<OT>::SolType& values) const {
		if constexpr (OT::total_order) {
			auto& sol = lb.solution;
			OT::Subtract(sol, values, sol);
			return lb;
		} else {
			for (size_t i = 0; i < lb->Size(); i++) {
				auto& sol = lb->GetMutable(i).solution;
				OT::Subtract(sol, values, sol);
			}
			auto new_lb = InitializeSol<OT>();
			AddSols<OT>(new_lb, lb);
			return new_lb;
		}
	}

	template <class OT>
	PairLowerBoundOptimal<OT> SimilarityLowerBoundComputer<OT>::ComputeLowerBound(ADataView& data, const Branch& branch, int depth, int num_nodes, Cache<OT>* cache) {
		auto empty_lb = InitializeSol<OT>(true);
		if (disabled_) { return PairLowerBoundOptimal<OT>(empty_lb, false); }

		const double max_difference = 1.0; // 1.0 is a dummy parameter. If a dataset differs more than 100%, dont compute a similarity bound. Todo: add configurable parameter
		PairLowerBoundOptimal<OT> result(empty_lb, false);
		auto& lb = result.lower_bound;
		for (auto& entry : archive_[depth]) {
			if (entry.data.Size() > data.Size() * (1 + max_difference) || entry.data.Size() < data.Size() * (1 - max_difference)) { continue; }

			if constexpr (OT::custom_similarity_lb) {
				auto metrics = task->ComputeSimilarityLowerBound(entry.data, data);
				if (metrics.total_difference > max_difference * data.Size()) continue;

				auto entry_lower_bound = cache->RetrieveLowerBound(entry.data, entry.branch, depth, num_nodes);
				entry_lower_bound = SubstractLB(entry_lower_bound, metrics.subtract);

				if (metrics.total_difference == 0) {
					cache->TransferAssignmentsForEquivalentBranches(entry.data, entry.branch, data, branch);
					if (cache->IsOptimalAssignmentCached(data, branch, depth, num_nodes)) {
						result.optimal = true;
						result.lower_bound = entry_lower_bound;
						break;
					}
				}
				AddSolsInv<OT>(lb, entry_lower_bound);
			} else {
				DifferenceMetrics metrics = BinaryDataDifferenceComputer::ComputeDifferenceMetrics(entry.data, data);
				if (metrics.total_difference > max_difference * data.Size()) continue;

				auto entry_lower_bound = cache->RetrieveLowerBound(entry.data, entry.branch, depth, num_nodes);
				SolType subtract = _worst[0] * metrics.num_removals[0];
				for (int k = 1; k < data.NumLabels(); k++)
					OT::Add(subtract, _worst[k] * metrics.num_removals[k], subtract);
				entry_lower_bound = SubstractLB(entry_lower_bound, subtract);

				if (metrics.total_difference == 0) {
					cache->TransferAssignmentsForEquivalentBranches(entry.data, entry.branch, data, branch);
					if (cache->IsOptimalAssignmentCached(data, branch, depth, num_nodes)) {
						result.optimal = true;
						result.lower_bound = entry_lower_bound;
						break;
					}
				}
				AddSolsInv<OT>(lb, entry_lower_bound);
			}
		}
		return result;
	}

	template <class OT>
	void SimilarityLowerBoundComputer<OT>::UpdateArchive(ADataView& data, const Branch& branch, int depth) {
		if (disabled_) { return; }

		SimilarityLowerBoundComputer<OT>::ArchiveEntry entry (data, branch);
		if (archive_[depth].size() < 2) { // TODO test with different values for this. What if larger values? More aggressive pruning might be useful
			archive_[depth].push_back(entry);
		} else {
			GetMostSimilarStoredData(data, depth) = entry;
		}
	}

	template <class OT>
	void SimilarityLowerBoundComputer<OT>::Initialise(OT* optimization_task, int num_labels, int max_depth, int size) {
		if (disabled_) { return; }

		task = optimization_task;
		archive_.resize(max_depth + 1);

		//_best_zero = true;
		_worst.resize(num_labels);
		for (int k = 0; k < num_labels ; k++) {
			//_best[k] = optimization_task->GetInstanceLowerBound(k);
			_worst[k] = optimization_task->GetWorstPerLabel(k);
			//if (_best[k] != 0) _best_zero = false;
		}
		//if (!_best_zero) Disable(); // todo: include addition of added data points * best
		
	}

	template <class OT>
	void SimilarityLowerBoundComputer<OT>::Disable() {
		disabled_ = true;
	}

	template <class OT>
	typename SimilarityLowerBoundComputer<OT>::ArchiveEntry& SimilarityLowerBoundComputer<OT>::GetMostSimilarStoredData(ADataView& data, int depth) {
		runtime_assert(archive_[depth].size() > 0);

		SimilarityLowerBoundComputer<OT>::ArchiveEntry* best_entry = NULL;
		int best_similiarity_score = INT32_MAX;
		for (auto& archieve_entry : archive_[depth]) {
			int similiarity_score = BinaryDataDifferenceComputer::ComputeDifferenceMetrics(archieve_entry.data, data).total_difference;
			if (similiarity_score < best_similiarity_score) {
				best_entry = &archieve_entry;
				best_similiarity_score = similiarity_score;
			}
		}
		runtime_assert(best_similiarity_score < INT32_MAX);
		return *best_entry;
	}


	template class SimilarityLowerBoundComputer<Accuracy>;
	template class SimilarityLowerBoundComputer<CostComplexAccuracy>;

	template class SimilarityLowerBoundComputer<Regression>;
	template class SimilarityLowerBoundComputer<CostComplexRegression>;
	template class SimilarityLowerBoundComputer<SimpleLinearRegression>;
	template class SimilarityLowerBoundComputer<PieceWiseLinearRegression>;

	template class SimilarityLowerBoundComputer<CostSensitive>;
	template class SimilarityLowerBoundComputer<InstanceCostSensitive>;
	template class SimilarityLowerBoundComputer<F1Score>;
	template class SimilarityLowerBoundComputer<GroupFairness>;
	template class SimilarityLowerBoundComputer<EqOpp>;
	template class SimilarityLowerBoundComputer<PrescriptivePolicy>;
}