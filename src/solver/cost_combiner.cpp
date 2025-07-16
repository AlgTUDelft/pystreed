#include "solver/terminal_solver.h"

namespace STreeD {

	template <class OT>
	CostCalculator<OT>::CostCalculator(OT* task, int num_features, int num_labels, const std::vector<int>& feature_order) : num_features(num_features), 
		task(task), counter(num_features), cost_storage(num_labels, CostStorage<OT>(num_features)), num_nodes(-1),
		branching_costs(num_features, std::vector<typename CostCalculator<OT>::BranchSolD2Type>(num_features)),
		index_infos(num_features, std::vector<IndexInfo>(num_features)), feature_order(feature_order) {
		InitializeIndexInfos(num_features);
	}

	template <class OT>
	bool CostCalculator<OT>::Initialize(const ADataView& data, const typename CostCalculator<OT>::Context& context, int num_nodes) {
		const bool depth_same = (num_nodes == 1) == (this->num_nodes == 1);
		const bool using_incremental_updates = this->data.IsInitialized() && depth_same;
		//ADataView data_add(data.GetData(), data.NumLabels()), data_remove(data.GetData(), data.NumLabels());
		if (using_incremental_updates) {
			data_add.ResetReserve(data);
			data_remove.ResetReserve(data);
			BinaryDataDifferenceComputer::ComputeDifference(this->data, data, data_add, data_remove);
			if (data_add.Size() == 0 && data_remove.Size() == 0 && (num_nodes == this->num_nodes || !OT::terminal_filter)) return false;
		}
		
		this->data = data;
		this->num_nodes = num_nodes;

		if (using_incremental_updates && data_add.Size() + data_remove.Size() < data.Size()) {
			UpdateCosts(data_add, +1);
			UpdateCosts(data_remove, -1);
		} else {
			for (size_t k = 0; k < cost_storage.size(); k++)
				cost_storage[k].ResetToZeros();
			counter.ResetToZeros();
			
			UpdateCosts(data, +1);
		}
		ResetBranchingCosts(); // TODO recompute even if no changes made, because context changes
		UpdateBranchingCosts(data, context); // TODO recompute even if no changes made, because context changes
		return true;
	}

	template <class OT>
	void CostCalculator<OT>::InitializeReconstruct(const ADataView& data, const typename CostCalculator<OT>::Context& context, int feature) {
		for (size_t k = 0; k < cost_storage.size(); k++)
			cost_storage[k].ResetToZerosReconstruct(feature);
		counter.ResetToZeros();
		UpdateCostsReconstruct(data, feature);
		ResetBranchingCosts();
		UpdateBranchingCosts(data, context);
		this->data = ADataView(); // Reset data, so that incremental updates are not used in the next normal call
	}

	template <class OT>
	void CostCalculator<OT>::InitializeIndexInfos(int num_features) {
		for (int feature1 = 0; feature1 < num_features; feature1++) {
			for (int feature2 = 0; feature2 < num_features; feature2++) {
				int f1 = feature1;
				int f2 = feature2;
				auto& index_info = index_infos[f1][f2];
				index_info.equal = f1 == f2;
				index_info.swap = f1 > f2;
				if (index_info.swap) std::swap(f1, f2);
				index_info.ix_f1f1 = num_features * f1 + f1 - f1 * (f1 + 1) / 2;
				index_info.ix_f1f2 = num_features * f1 + f2 - f1 * (f1 + 1) / 2;
				index_info.ix_f2f2 = num_features * f2 + f2 - f2 * (f2 + 1) / 2;
			}
		}
	}

	template <class OT>
	int CostCalculator<OT>::ProbeDifference(const ADataView& data) const {
		return this->data.IsInitialized()
			? BinaryDataDifferenceComputer::ComputeDifferenceMetrics(this->data, data).total_difference
			: -1;
	}


	template <class OT, bool update_count, bool update_cost> void UpdateCountCost(
		const AInstance* data_point,
		CostStorage<OT>& _cost_storage,
		Counter& counter,
		typename OT::SolD2Type& costs,
		int multiplier,
		bool only_one_dimension) {
		
		if constexpr (OT::use_weights) {
			multiplier *= int(data_point->GetWeight());
		}

		const int num_present_features = data_point->NumPresentFeatures();
		if constexpr (update_cost) {
			_cost_storage.UpdateTotalCosts(costs);
		}

		if (only_one_dimension) {
			for (int i = 0; i < num_present_features; i++) {
				const int feature1 = data_point->GetJthPresentFeature(i);
				if constexpr (update_cost) {
					_cost_storage.UpdateCosts(feature1, feature1, costs);
				}
				if constexpr (update_count) {
					counter.UpdateCount(feature1, feature1, multiplier);
				}
			}
			return;
		}
		
		for (const int& index: data_point->GetPresentFeaturePairIndices()) {
			if constexpr (update_cost) {
				_cost_storage.UpdateCosts(index, costs);
			}
			if constexpr (update_count) {
				counter.UpdateCount(index, multiplier);
			}
		}

	}

	template <class OT>
	void CostCalculator<OT>::UpdateCosts(const ADataView& data, int multiplier) {
		typename CostCalculator<OT>::SolD2Type costs; // dummy values
		const bool only_one_dimension = num_nodes == 1;
		labels = { 0 };
		for (int org_label = 0; org_label < data.NumLabels(); org_label++) {
			if (data.NumLabels() > 1) {
				labels.clear();
				for (int label = 0; label < data.NumLabels(); label++) {
					if (!OT::terminal_zero_costs_true_label || label != org_label) labels.push_back(label);
				}
			} 
			int first_label = labels[0];
			for (auto& data_point : data.GetInstancesForLabel(org_label)) {
				
				for(const int label: labels) {

					auto& _cost_storage = cost_storage[label];
					task->GetInstanceLeafD2Costs(data_point, org_label, label, costs, multiplier);
					if (task->IsD2ZeroCost(costs)) { // Zero costs
						if (label > first_label) continue;
						UpdateCountCost<OT, true, false>(data_point, _cost_storage, counter, costs, multiplier, only_one_dimension);
					} else if (label > first_label) {
						UpdateCountCost<OT, false, true>(data_point, _cost_storage, counter, costs, multiplier, only_one_dimension);
					} else {
						UpdateCountCost<OT, true, true>(data_point, _cost_storage, counter, costs, multiplier, only_one_dimension);
					}
				}
				if constexpr (OT::use_weights) {
					counter.UpdateTotalCount(int(data_point->GetWeight() * multiplier));
				}
			}
			

		}
		if constexpr (!OT::use_weights) {
			counter.UpdateTotalCount(int(data.Size() * multiplier));
		}
	}

	template <class OT>
	void CostCalculator<OT>::UpdateCostsReconstruct(const ADataView& data, int feature) {
		typename CostCalculator<OT>::SolD2Type costs; // dummy values
		for (int org_label = 0; org_label < data.NumLabels(); org_label++) {
			for (auto& data_point : data.GetInstancesForLabel(org_label)) {

				const bool feature_is_present = data_point->IsFeaturePresent(feature);
				const int num_present_features = data_point->NumPresentFeatures();
				for (int label = 0; label < data.NumLabels(); label++) {

					auto& _cost_storage = cost_storage[label];
					task->GetInstanceLeafD2Costs(data_point, org_label, label, costs, +1);
					_cost_storage.UpdateTotalCosts(costs);

					if (task->IsD2ZeroCost(costs)) continue; // zero costs, don't add
					
					for (int j = 0; j < num_present_features; j++) {
						int feature2 = data_point->GetJthPresentFeature(j);
						_cost_storage.UpdateCosts(feature2, feature2, costs);
					}
					if (!feature_is_present) continue;
					for (int j = 0; j < num_present_features; j++) {
						int feature1 = feature;
						int feature2 = data_point->GetJthPresentFeature(j);
						if (feature1 == feature2) continue;
						if (feature1 > feature2) std::swap(feature1, feature2);
						_cost_storage.UpdateCosts(feature1, feature2, costs);
					}
				}
				
				int weight = int(data_point->GetWeight());
				counter.UpdateTotalCount(weight);
				for (int j = 0; j < num_present_features; j++) {
					int feature2 = data_point->GetJthPresentFeature(j);
					counter.UpdateCount(feature2, feature2, weight);
				}
				if (!feature_is_present) continue;
				for (int j = 0; j < num_present_features; j++) {
					int feature1 = feature;
					int feature2 = data_point->GetJthPresentFeature(j);
					if (feature1 == feature2) continue;
					if (feature1 > feature2) std::swap(feature1, feature2);
					counter.UpdateCount(feature1, feature2, weight);
				}
			}

		}
		
	}

	template <class OT>
	void CostCalculator<OT>::UpdateBranchingCosts(const ADataView& data, const typename CostCalculator<OT>::Context& context) {
		if constexpr (OT::has_branching_costs) {

			if constexpr (OT::element_branching_costs) {
				runtime_assert(1 == 0);// Not implemented yet
			} else {
				typename CostCalculator<OT>::Context sub_context;
				for (int f1 = 0; f1 < data.NumFeatures(); f1++) {
					task->GetLeftContext(data, context, f1, sub_context); //Ignore left/right and only look at the feature that is checked
					for (int f2 = 0; f2 < data.NumFeatures(); f2++) {
						if (f1 == f2) continue;
						branching_costs[f1][f2] = task->GetBranchingCosts(sub_context, f2);
					}
					branching_costs[f1][f1] = task->GetBranchingCosts(context, f1);
				}
			}
		}
	}

	template <class OT>
	void CostCalculator<OT>::ResetBranchingCosts() {
		if constexpr (OT::has_branching_costs) {
			int num_features = int(branching_costs.size());
			for (int i = 0; i < num_features; i++) {
				for (int j = 0; j < num_features; j++) {
					branching_costs[i][j] = typename OT::BranchSolD2Type(); // must have a default constructor
				}
			}
		}
	}

	template <class OT>
	const typename CostCalculator<OT>::SolType CostCalculator<OT>::GetBranchingCosts(int f1) const {
		if constexpr (OT::has_branching_costs) {
			return task->ComputeD2BranchingCosts(branching_costs[f1][f1], counter.GetTotalCount());
		} else {
			return CostCalculator<OT>::SolType();
		}
	}

	template <class OT>
	const typename CostCalculator<OT>::SolType CostCalculator<OT>::GetBranchingCosts0(int count0, int f1, int f2) const {
		if constexpr (OT::has_branching_costs) {
			return task->ComputeD2BranchingCosts(branching_costs[f1][f2], count0);
		} else {
			return CostCalculator<OT>::SolType();
		}
	}

	template <class OT>
	const typename CostCalculator<OT>::SolType CostCalculator<OT>::GetBranchingCosts1(int count1, int f1, int f2) const {
		if constexpr (OT::has_branching_costs) {
			return task->ComputeD2BranchingCosts(branching_costs[f1][f2], count1);
		} else {
			return CostCalculator<OT>::SolType();
		}
	}

	template <class OT>
	void CostCalculator<OT>::CalcLeafSol(SolType& sol, int label, SolLabelType& label_out) const {
		const auto& _cost_storage = cost_storage[label];
		const auto& total_costs = _cost_storage.GetTotalCosts();
		int count = counter.GetTotalCount();
		task->ComputeD2Costs(total_costs, count, sol);
		if constexpr (OT::custom_get_label || std::is_same<typename OT::LabelType, double>::value) {
			label_out = GetLabel(label, total_costs, count);
		} else {
			label_out = label;
		}
	}

	template <class OT>
	void CostCalculator<OT>::CalcSols(const Counts& counts, Sols<OT>& sols, int label, int f1, int f2) const {
		bool swap = f1 > f2;
		if (swap) std::swap(f1, f2);
		const auto& _cost_storage = cost_storage[label];
		const auto& total_costs = _cost_storage.GetTotalCosts();
		const auto& costsf1f2 = _cost_storage.GetCosts(f1, f2);
		const auto& costsf1f1 = _cost_storage.GetCosts(f1, f1);
		const auto& costsf2f2 = _cost_storage.GetCosts(f2, f2);

		if (f1 == f2) {
			task->ComputeD2Costs(total_costs - costsf1f2, counts.count00, sols.sol00);
			task->ComputeD2Costs(costsf1f2, counts.count11, sols.sol11);
			return;
		}
		temp_costs1 = total_costs;
		temp_costs1 += costsf1f2;
		temp_costs1 -= costsf1f1;
		temp_costs1 -= costsf2f2;
		//task->ComputeD2Costs((total_costs + costsf1f2) - costsf1f1 - costsf2f2 , counts.count00, sols.sol00);
		task->ComputeD2Costs(temp_costs1, counts.count00, sols.sol00);
		task->ComputeD2Costs(costsf1f2, counts.count11, sols.sol11);
		if (swap) {
			task->ComputeD2Costs(costsf2f2 - costsf1f2, counts.count10, sols.sol10);
			task->ComputeD2Costs(costsf1f1 - costsf1f2, counts.count01, sols.sol01);
			return;
		}
		task->ComputeD2Costs(costsf2f2 - costsf1f2, counts.count01, sols.sol01);
		task->ComputeD2Costs(costsf1f1 - costsf1f2, counts.count10, sols.sol10);
	}

	template <class OT>
	void CostCalculator<OT>::CalcSols(const Counts& counts, Sols<OT>& sols, int label, const IndexInfo& index) const {
		const auto& _cost_storage = cost_storage[label];
		const auto& total_costs = _cost_storage.GetTotalCosts();
		const auto& costsf1f2 = _cost_storage.GetCosts(index.ix_f1f2);
		const auto& costsf1f1 = _cost_storage.GetCosts(index.ix_f1f1);
		const auto& costsf2f2 = _cost_storage.GetCosts(index.ix_f2f2);

		if (index.equal) {
			task->ComputeD2Costs(total_costs - costsf1f2, counts.count00, sols.sol00);
			task->ComputeD2Costs(costsf1f2, counts.count11, sols.sol11);
			return;
		}
		// Temp_costs2 = costs for f1, but not f2
		temp_costs2 = costsf1f1;
		temp_costs2 -= costsf1f2;

		// Temp costs1 = costs for neither f1 nor f2
		temp_costs1 = total_costs;
		temp_costs1 -= temp_costs2;
		temp_costs1 -= costsf2f2;
		task->ComputeD2Costs(temp_costs1, counts.count00, sols.sol00);
		task->ComputeD2Costs(costsf1f2, counts.count11, sols.sol11);
		if (index.swap) {
			task->ComputeD2Costs(costsf2f2 - costsf1f2, counts.count10, sols.sol10);
			task->ComputeD2Costs(costsf1f1 - costsf1f2, counts.count01, sols.sol01);
			return;
		}
		// Temp costs1 = costs for f2 but not f1
		temp_costs1 = costsf2f2;
		temp_costs1 -= costsf1f2;
		task->ComputeD2Costs(temp_costs1, counts.count01, sols.sol01);
		// temp costs2 is still costs for f1 but not f2
		task->ComputeD2Costs(temp_costs2, counts.count10, sols.sol10);
	}

	template <class OT>
	void CostCalculator<OT>::CalcSol00(typename CostCalculator<OT>::SolType& sol, int label, int f1, int f2) const {
		const auto& _cost_storage = cost_storage[label];
		const auto& total_costs = _cost_storage.GetTotalCosts();
		if (f1 == f2) {
			const auto& costs = _cost_storage.GetCosts(f1, f2);
			task->ComputeD2Costs(total_costs - costs, GetCount00(f1, f2), sol);
		} else {
			if (f1 > f2) std::swap(f1, f2);
			const auto& costsf1f2 = _cost_storage.GetCosts(f1, f2);
			const auto& costsf1f1 = _cost_storage.GetCosts(f1, f1);
			const auto& costsf2f2 = _cost_storage.GetCosts(f2, f2);
			task->ComputeD2Costs(total_costs + costsf1f2 - costsf1f1 - costsf2f2 , GetCount00(f1, f2), sol);
		}
	}

	template <class OT>
	void CostCalculator<OT>::CalcSol11(typename CostCalculator<OT>::SolType& sol, int label, int f1, int f2) const {
		if (f1 > f2) std::swap(f1, f2);
		task->ComputeD2Costs(cost_storage[label].GetCosts(f1, f2), GetCount11(f1, f2), sol);
	}

	template <class OT>
	const typename CostCalculator<OT>::SolD2Type CostCalculator<OT>::GetCosts00(int label, int f1, int f2) const {
		if (f1 == f2)
			return cost_storage[label].GetTotalCosts() - GetCosts11(label, f1, f1);
		else if (f1 > f2) { std::swap(f1, f2); }
		return cost_storage[label].GetTotalCosts() + GetCosts11(label, f1, f2) - GetCosts11(label, f1, f1) - GetCosts11(label, f2, f2);
	}

	template <class OT>
	const typename CostCalculator<OT>::SolD2Type CostCalculator<OT>::GetCosts01(int label, int f1, int f2) const {
		if (f1 > f2) { return GetCosts10(label, f2, f1); }
		return cost_storage[label].GetCosts(f2, f2) - cost_storage[label].GetCosts(f1, f2);
	}

	template <class OT>
	const typename CostCalculator<OT>::SolD2Type CostCalculator<OT>::GetCosts10(int label, int f1, int f2) const {
		if (f1 > f2) { return GetCosts01(label, f2, f1); }
		return cost_storage[label].GetCosts(f1, f1) - cost_storage[label].GetCosts(f1, f2);
	}

	template <class OT>
	const typename CostCalculator<OT>::SolD2Type CostCalculator<OT>::GetCosts11(int label, int f1, int f2) const {
		if (f1 > f2) { std::swap(f1, f2); }
		return cost_storage[label].GetCosts(f1, f2);
	}

	template <class OT>
	void CostCalculator<OT>::GetCounts(Counts& counts, int f1, int f2) const {
		bool swap = f1 > f2;
		if (swap) std::swap(f1, f2);
		int countf1f1 = counter.GetCount(f1, f1);
		int countf1f2 = counter.GetCount(f1, f2);
		int countf2f2 = counter.GetCount(f2, f2);
		counts.count00 = counter.GetTotalCount() - countf1f1 - countf2f2 + countf1f2;
		counts.count11 = countf1f2;
		if (swap) {
			counts.count10 = countf2f2 - countf1f2;
			counts.count01 = countf1f1 - countf1f2;
			return;
		}
		counts.count01 = countf2f2 - countf1f2;
		counts.count10 = countf1f1 - countf1f2;
	}

	template <class OT>
	void CostCalculator<OT>::GetCounts(Counts& counts, const IndexInfo& index) const {
		int countf1f1 = counter.GetCount(index.ix_f1f1);
		int countf2f2 = counter.GetCount(index.ix_f2f2);
		counts.count11 = counter.GetCount(index.ix_f1f2);
		counts.count00 = counter.GetTotalCount() - countf1f1 - countf2f2 + counts.count11;
		if (index.swap) {
			counts.count10 = countf2f2 - counts.count11;
			counts.count01 = countf1f1 - counts.count11;
			return;
		}
		counts.count01 = countf2f2 - counts.count11;
		counts.count10 = countf1f1 - counts.count11;
	}

	template <class OT>
	int CostCalculator<OT>::GetCount00(int f1, int f2) const {
		if (f1 > f2) { std::swap(f1, f2); }
		return counter.GetTotalCount() - (GetCount11(f1, f1) + GetCount11(f2, f2) - GetCount11(f1, f2));
	}

	template <class OT>
	int CostCalculator<OT>::GetCount01(int f1, int f2) const {
		if (f1 > f2) { return GetCount10(f2, f1); }
		return counter.GetCount(f2, f2) - counter.GetCount(f1, f2);
	}

	template <class OT>
	int CostCalculator<OT>::GetCount10(int f1, int f2) const {
		if (f1 > f2) { return GetCount01(f2, f1); }
		return counter.GetCount(f1, f1) - counter.GetCount(f1, f2);
	}

	template <class OT>
	int CostCalculator<OT>::GetCount11(int f1, int f2) const {
		if (f1 > f2) { std::swap(f1, f2); }
		return counter.GetCount(f1, f2);
	}

	template <class OT>
	const typename CostCalculator<OT>::SolLabelType CostCalculator<OT>::GetLabel00(int label, int f1, int f2) const {
		if constexpr (OT::custom_get_label || std::is_same<typename OT::LabelType, double>::value) {
			return task->GetLabel(GetCosts00(label, f1, f2), GetCount00(f1, f2));
		} else {
			return label;
		}
	}

	template <class OT>
	const typename CostCalculator<OT>::SolLabelType CostCalculator<OT>::GetLabel01(int label, int f1, int f2) const {
		if constexpr (OT::custom_get_label || std::is_same<typename OT::LabelType, double>::value) {
			return task->GetLabel(GetCosts01(label, f1, f2), GetCount01(f1, f2));
		} else {
			return label;
		}
	}

	template <class OT>
	const typename CostCalculator<OT>::SolLabelType CostCalculator<OT>::GetLabel10(int label, int f1, int f2) const {
		if constexpr (OT::custom_get_label || std::is_same<typename OT::LabelType, double>::value) {
			return task->GetLabel(GetCosts10(label, f1, f2), GetCount10(f1, f2));
		} else {
			return label;
		}
	}

	template <class OT>
	const typename CostCalculator<OT>::SolLabelType CostCalculator<OT>::GetLabel11(int label, int f1, int f2) const {
		if constexpr (OT::custom_get_label || std::is_same<typename OT::LabelType, double>::value) {
			return task->GetLabel(GetCosts11(label, f1, f2), GetCount11(f1, f2));
		} else {
			return label;
		}
	}

	template <class OT>
	const typename CostCalculator<OT>::SolLabelType CostCalculator<OT>::GetLabel(int label, const typename CostCalculator<OT>::SolD2Type& costs, int count) const {
		if constexpr (OT::custom_get_label || std::is_same<typename OT::LabelType, double>::value) {
			return task->GetLabel(costs, count);
		} else {
			return label;
		}
	}

	template class CostCalculator<Accuracy>;
	template class CostCalculator<CostComplexAccuracy>;
	template class CostCalculator<BalancedAccuracy>;

	template class CostCalculator<Regression>;
	template class CostCalculator<CostComplexRegression>;
	template class CostCalculator<SimpleLinearRegression>;

	template class CostCalculator<CostSensitive>;
	template class CostCalculator<InstanceCostSensitive>;
	template class CostCalculator<F1Score>;
	template class CostCalculator<GroupFairness>;

	template class CostCalculator<EqOpp>;
	template class CostCalculator<PrescriptivePolicy>;
	template class CostCalculator<SurvivalAnalysis>;
	
}