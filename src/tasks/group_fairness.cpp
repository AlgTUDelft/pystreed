#include "tasks/group_fairness.h"

namespace STreeD {

	void GroupFairness::InformTrainData(const ADataView& train_data, const DataSummary& train_summary) {
		OptimizationTask::InformTrainData(train_data, train_summary);
		train_group0_size = 0;
		train_group1_size = 0;
		for (int label = 0; label < train_data.NumLabels(); label++) {
			for (auto& dp : train_data.GetInstancesForLabel(label)) {
				if (dp->IsFeaturePresent(0))
					train_group1_size++;
				else
					train_group0_size++;
			}
		}
	}

	void GroupFairness::InformTestData(const ADataView& test_data, const DataSummary& test_summary) {
		OptimizationTask::InformTestData(test_data, test_summary);
		test_group0_size = 0;
		test_group1_size = 0;
		for (int label = 0; label < test_data.NumLabels(); label++) {
			for (auto& dp : test_data.GetInstancesForLabel(label)) {
				if (dp->IsFeaturePresent(0))
					test_group1_size++;
				else
					test_group0_size++;
			}
		}
	}

	GroupFairnessSol GroupFairness::GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const {
		runtime_assert(data.NumLabels() == 2);
		int group0_size = 0;
		for (int k = 0; k < data.NumLabels(); k++) {
			for (auto& dp : data.GetInstancesForLabel(k)) {
				group0_size += dp->IsFeaturePresent(0) ? 0 : 1; // Zeroth feature is the sensitive feature
			}
		}
		return label ?
			GroupFairnessSol({
				data.NumInstancesForLabel(0), // Binary misclassification score
				group0_size / double(train_group0_size),
				(data.Size() - group0_size) / double(train_group1_size)
				}) :
			GroupFairnessSol({
				data.NumInstancesForLabel(1), // Binary misclassification score
				(data.Size() - group0_size) / double(train_group1_size),
				group0_size / double(train_group0_size)
				});
	}

	void GroupFairness::GetInstanceLeafD2Costs(const AInstance* instance, int org_label, int label, GroupFairnessSol& costs, int multiplier) const {
		int group0 = instance->IsFeaturePresent(0) ? 0 : 1;
		costs = label ?
			GroupFairnessSol({
				multiplier * (label - org_label), // Binary misclassification score
				multiplier * (group0 / double(train_group0_size)),
				multiplier * ((1 - group0) / double(train_group1_size))
				}) :
			GroupFairnessSol({
				multiplier * (org_label - label), // Binary misclassification score
				multiplier * ((1 - group0) / double(train_group1_size)),
				multiplier * (group0 / double(train_group0_size))
				});
	}


	double GroupFairness::ComputeTrainScore(const GroupFairnessSol& train_value) const {
		runtime_assert(train_summary.instances_per_class.size() == 2);
		double disc = std::max(train_value.group0_score, train_value.group1_score) - 1;
		if (disc <= discrimination_limit)
			return ((double)(train_summary.size - train_value.misclassifications)) / ((double)train_summary.size);
		else
			return 0;
	}

	double GroupFairness::ComputeTrainTestScore(const GroupFairnessSol& train_value) const {
		return ComputeTrainScore(train_value);
	}

	double GroupFairness::ComputeTestTestScore(const GroupFairnessSol& test_value) const {
		runtime_assert(test_summary.instances_per_class.size() == 2);
		
		double disc = std::max(test_value.group0_score, test_value.group1_score) - 1;
		if (disc <= discrimination_limit)
			return ((double)(test_summary.size - test_value.misclassifications)) / ((double)test_summary.size);
		else
			return 0;
	}

	void GroupFairness::RelaxRootSolution(Node<GroupFairness>& sol) const {
		//sol.solution.group0_score = 0;
		//sol.solution.group1_score = 0;
		sol.solution.root_solution = true;
	}

}