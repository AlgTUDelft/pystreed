#include "tasks/eq_opp.h"

namespace STreeD {

	void EqOpp::InformTrainData(const ADataView& train_data, const DataSummary& train_summary) {
		OptimizationTask::InformTrainData(train_data, train_summary);
		runtime_assert(train_data.NumLabels() == 2);
		// only count instances with label = 1 (positive label)
		train_group0_size = 0;
		train_group1_size = 0;
		for (auto& dp : train_data.GetInstancesForLabel(1)) {
			if (dp->IsFeaturePresent(0))
				train_group1_size++;
			else
				train_group0_size++;
		}
	}

	void EqOpp::InformTestData(const ADataView& test_data, const DataSummary& test_summary) {
		OptimizationTask::InformTestData(test_data, test_summary);
		// only count instances with label = 1 (positive label)
		test_group0_size = 0;
		test_group1_size = 0;
		if (test_data.NumLabels() < 2) return;
		for (auto& dp : test_data.GetInstancesForLabel(1)) {
			if (dp->IsFeaturePresent(0))
				test_group1_size++;
			else
				test_group0_size++;
		}
	}

	EqOppSol EqOpp::GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const {
		runtime_assert(data.NumLabels() == 2);
		int group0_size = 0;
		for (auto& dp : data.GetInstancesForLabel(1)) {
			group0_size += dp->IsFeaturePresent(0) ? 0 : 1; // Zeroth feature is the sensitive feature
		}
		return label ?
			EqOppSol({
				data.NumInstancesForLabel(0), // Binary misclassification score
				group0_size / double(train_group0_size),
				(data.NumInstancesForLabel(1) - group0_size) / double(train_group1_size)
				}) :
			EqOppSol({
				data.NumInstancesForLabel(1), // Binary misclassification score
				(data.NumInstancesForLabel(1) - group0_size) / double(train_group1_size),
				group0_size / double(train_group0_size)
				});
	}

	void EqOpp::GetInstanceLeafD2Costs(const AInstance* instance, int org_label, int label, EqOppSol& costs, int multiplier) const {
		int label1 = org_label == 1 ? 1 : 0;
		int group0 = instance->IsFeaturePresent(0) ? 0 : 1;
		int group1 = org_label == 1 ? 1 - group0 : 0;
		group0 = org_label == 1 ? group0 : 0;
		costs = label ?
			EqOppSol({
				multiplier * (label - org_label), // Binary misclassification score
				multiplier * (group0 / double(train_group0_size)),
				multiplier * (group1 / double(train_group1_size))
				}) :
			EqOppSol({
				multiplier * (org_label - label), // Binary misclassification score
				multiplier * (group1 / double(train_group1_size)),
				multiplier * (group0 / double(train_group0_size))
				});
	}


	double EqOpp::ComputeTrainScore(const EqOppSol& train_value) const {
		runtime_assert(train_summary.instances_per_class.size() == 2);
		double disc = std::max(train_value.group0_score, train_value.group1_score) - 1;
		if (disc <= discrimination_limit)
			return ((double)(train_summary.size - train_value.misclassifications)) / ((double)train_summary.size);
		else
			return 0;
	}

	double EqOpp::ComputeTrainTestScore(const EqOppSol& train_value) const {
		return ComputeTrainScore(train_value);
	}

	double EqOpp::ComputeTestTestScore(const EqOppSol& test_value) const {
		runtime_assert(test_summary.instances_per_class.size() == 2);
		
		double disc = std::max(test_value.group0_score, test_value.group1_score) - 1;
		if (disc <= discrimination_limit)
			return ((double)(test_summary.size - test_value.misclassifications)) / ((double)test_summary.size);
		else
			return 0;
	}

	void EqOpp::RelaxRootSolution(Node<EqOpp>& sol) const {
		sol.solution.root_solution = true;
	}

}