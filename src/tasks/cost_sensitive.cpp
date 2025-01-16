#include "tasks/cost_sensitive.h"

namespace STreeD {

	std::string FeatureCostSpecifier::ToString() const  {
		std::ostringstream s;
		s << "Feature cost: " << feature_cost 
			<< ", Discount cost: " << discount_cost
			<< ", Group name: " << group_name
			<< ", Binary range: [" << binary_begin << ", " << binary_end << "]";
		return s.str();
	}

	CostSpecifier::CostSpecifier(const std::string& cost_filename, int num_labels) {
		std::ifstream file(cost_filename.c_str());
		if (!file) { std::cout << "Error: File " << cost_filename << " does not exist!\n"; runtime_assert(file); }
		std::string line;
		size_t line_no = -1;
		
		misclassification_costs = std::vector<std::vector<double>>(num_labels, std::vector<double>(num_labels));
		std::vector<FeatureCostSpecifier> feature_cost_specifiers;
		while (std::getline(file, line)) {
			line_no++;
			std::istringstream iss(line);
			double temp;
			if (line_no < num_labels) {
				for (int i = 0; i < num_labels; i++) {
					iss >> temp;
					misclassification_costs[line_no][i] = temp;
				}
			} else if (line_no == num_labels) {
				continue; // skip column header line
				//columns are described by: Attribute name / test cost / discount / group / binarize begin (ix) / binarize end (ix)
			} else {
				std::string attribute_name;
				double test_cost;
				double discount_cost;
				std::string discount_group;
				int binarize_begin_ix;
				int binarize_end_ix;
				iss >> attribute_name;
				iss >> test_cost;
				iss >> discount_cost;
				iss >> discount_group;
				iss >> binarize_begin_ix;
				iss >> binarize_end_ix;
				feature_cost_specifiers.emplace_back(test_cost, discount_cost, discount_group, binarize_begin_ix, binarize_end_ix);
			}
		}
		Initialize(feature_cost_specifiers);
	}

	CostSpecifier::CostSpecifier(const std::vector<std::vector<double>>& cost_matrix, const std::vector<FeatureCostSpecifier>& feature_costs) 
		: misclassification_costs(cost_matrix) {
		Initialize(feature_costs);
	}

	void CostSpecifier::Initialize(const std::vector<FeatureCostSpecifier>& feature_cost_specifiers) {
		int num_features = 0;
		for(auto& fcs: feature_cost_specifiers) {
			num_features += fcs.binary_end - fcs.binary_begin + 1;
		}		
		feature_costs = std::vector<double>(num_features);
		discount_costs = std::vector<double>(num_features);
		int fsquared = num_features * num_features;
		same_group = std::vector<int>(fsquared, 0);
		same_binarized = std::vector<int>(fsquared, 0);

		std::map<std::string, std::vector<int>> group_attribute_map;
		for(auto& fcs: feature_cost_specifiers) {
			if (!group_attribute_map.count(fcs.group_name)) group_attribute_map[fcs.group_name];
			runtime_assert(fcs.binary_begin <= fcs.binary_end);
			for (int f = fcs.binary_begin; f <= fcs.binary_end; f++) {
				if (f >= num_features) break;
				feature_costs[f] = fcs.feature_cost;
				discount_costs[f] = fcs.discount_cost;
				for (int f2 : group_attribute_map[fcs.group_name]) {
					same_group[f * num_features + f2] = true;
					same_group[f2 * num_features + f] = true;
				}
				group_attribute_map[fcs.group_name].push_back(f);
				for (int f2 = fcs.binary_begin; f2 <= fcs.binary_end; f2++) {
					if (f2 >= num_features) break;
					same_binarized[f * num_features + f2] = true;
					same_binarized[f2 * num_features + f] = true;
				}
			}
		}
		total_test_costs = ComputeTotalTestCosts();
		max_misclassification_costs = ComputeMaxMisclassificationCost();
		max_instance_costs = total_test_costs + max_misclassification_costs;
	}

	double CostSpecifier::ComputeTotalTestCosts() const {
		double total_cost = 0;
		int n_features = int(feature_costs.size());
		std::vector<bool> binary_done(n_features, false);
		std::vector<bool> group_done(n_features, false);
		for (int f = 0; f < feature_costs.size(); f++) {
			if (binary_done[f]) continue;
			else if (group_done[f]) {
				total_cost += discount_costs[f];
			} else {
				total_cost += feature_costs[f];
			}
			for (int g = f + 1; g < feature_costs.size(); g++) {
				if (same_binarized[f * n_features + g]) {
					binary_done[g] = true;
				}
				if (same_group[f * n_features + g]) {
					group_done[g] = true;
				}
			}
		}
		return total_cost;
	}

	double CostSpecifier::ComputeMaxMisclassificationCost() const {
		double max = -DBL_MAX;
		for (int i = 0; i < misclassification_costs.size();i++) {
			for (int j = 0; j < misclassification_costs[i].size(); j++) {
				if (misclassification_costs[i][j] > max) {
					max = misclassification_costs[i][j];
				}
			}
		}
		return max;
	}

	void CostSensitive::InformTrainData(const ADataView& train_data, const DataSummary& train_summary) {
		OptimizationTask::InformTrainData(train_data, train_summary);
		runtime_assert(cost_filename != "" || cost_specifier.IsInitialized());
		if(cost_filename != "") cost_specifier = CostSpecifier(cost_filename, train_data.NumLabels());
	}

	void CostSensitive::GetInstanceLeafD2Costs(const AInstance* instance, int org_label, int label, double& costs, int multiplier) const {
		costs = multiplier * cost_specifier.misclassification_costs[org_label][label];
	}

	double CostSensitive::GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const { 
		double error = 0;
		for (int k = 0; k < data.NumLabels(); k++) {
			if (k == label) continue;
			error += data.NumInstancesForLabel(k) * cost_specifier.misclassification_costs[k][label];
		}
		return error;
	}

	double CostSensitive::GetBranchingCosts(const BranchContext& context, int feature) const {
		int n_features = int(cost_specifier.feature_costs.size());
		for (int i = 0; i < context.GetBranch().Depth(); i++) {
			int f = context.GetBranch()[i] / 2;
			if (cost_specifier.same_binarized[f * n_features + feature]) return 0;
		}
		for (int i = 0; i < context.GetBranch().Depth(); i++) {
			int f = context.GetBranch()[i] / 2;
			if (cost_specifier.same_group[f * n_features + feature]) return cost_specifier.discount_costs[feature];
		}
		return cost_specifier.feature_costs[feature];

	}

	double ComputeScore(double value, double total_test_costs, double max_misclassification_costs, const DataSummary& data_summary) {
		double avg_cost = value / double(data_summary.size);
		int biggest_class = *std::max_element(std::begin(data_summary.instances_per_class), std::end(data_summary.instances_per_class));
		double inv_perc = 1.0 - (double(biggest_class) / double(data_summary.size));
		double standard_cost = total_test_costs + inv_perc * max_misclassification_costs;
		return { avg_cost / standard_cost };
	}

	double CostSensitive::ComputeTrainTestScore(double test_value) const {
		return ComputeScore(test_value, cost_specifier.total_test_costs, cost_specifier.max_misclassification_costs, train_summary);
	}

	double CostSensitive::ComputeTestTestScore(double test_value) const {
		return ComputeScore(test_value, cost_specifier.total_test_costs, cost_specifier.max_misclassification_costs, test_summary);
	}

}