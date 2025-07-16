#include "tasks/accuracy/balanced_accuracy.h"

namespace STreeD {

	void BalancedAccuracy::InformTrainData(const ADataView& train_data, const DataSummary& train_summary) {
		OptimizationTask::InformTrainData(train_data, train_summary);
		cost_matrix.resize(train_data.NumLabels(), 1);
		for (int label = 0; label < train_data.NumLabels(); label++) {
			for (int label2 = 0; label2 < train_data.NumLabels(); label2++) {
				if (label != label2) cost_matrix[label] *= train_data.NumInstancesForLabel(label2);
			}
		}
	}

	void BalancedAccuracy::InformTestData(const ADataView& test_data, const DataSummary& test_summary) {
		OptimizationTask::InformTestData(test_data, test_summary);
		test_cost_matrix.resize(test_data.NumLabels(), 1);
		for (int label = 0; label < test_data.NumLabels(); label++) {
			for (int label2 = 0; label2 < test_data.NumLabels(); label2++) {
				if (label != label2) test_cost_matrix[label] *= test_data.NumInstancesForLabel(label2);
			}
		}
	}

	int BalancedAccuracy::GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const { 
		int error = 0;
		for (int k = 0; k < data.NumLabels(); k++) {
			if (k == label) continue;
			error += data.NumInstancesForLabel(k) * cost_matrix[k];
		}
		return error;
	}

	int BalancedAccuracy::GetTestLeafCosts(const ADataView& data, const BranchContext& context, int label) const {
		int error = 0;
		for (int k = 0; k < data.NumLabels(); k++) {
			if (k == label) continue;
			int test_cost = test_cost_matrix.size() > 0 ? test_cost_matrix[k] : cost_matrix[k];
			error += data.NumInstancesForLabel(k) * test_cost;
		}
		return error;
	}

	// Compute the train score from the training solution value
	double BalancedAccuracy::ComputeTrainScore(int test_value) const {
		int max_cost = 0;
		for (int label = 0; label < train_summary.num_labels; label++)
			max_cost += train_summary.instances_per_class[label] * cost_matrix[label];
		return ((double)(max_cost - test_value)) / ((double)max_cost);
	}

	// Compute the test score on the training data from the test solution value
	double BalancedAccuracy::ComputeTrainTestScore(int test_value) const {
		int max_cost = 0;
		for (int label = 0; label < train_summary.num_labels; label++)
			max_cost += train_summary.instances_per_class[label] * cost_matrix[label];
		return ((double)(max_cost - test_value)) / ((double)max_cost);
	}

	// Compute the test score on the test data from the test solution value
	// (but still using the class division of the training data)
	double BalancedAccuracy::ComputeTestTestScore(int test_value) const {
		int max_cost = 0;
		for (int label = 0; label < test_summary.num_labels; label++)
			max_cost += test_summary.instances_per_class[label] * test_cost_matrix[label];
		return ((double)(max_cost - test_value)) / ((double)max_cost);
	}

}