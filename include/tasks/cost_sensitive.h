#pragma once
#include "tasks/optimization_task.h"

namespace STreeD {

	struct FeatureCostSpecifier {
		double feature_cost {0};
		double discount_cost {0};
		std::string group_name;
		int binary_begin {0};
		int binary_end {0};
		FeatureCostSpecifier(double feature_cost, double discount_cost, const std::string& group_name, int binary_begin, int binary_end) 
		 : feature_cost(feature_cost), discount_cost(discount_cost), group_name(group_name), binary_begin(binary_begin), binary_end(binary_end) {}
		std::string ToString() const;
	};

	struct CostSpecifier {
		CostSpecifier() = default;
		CostSpecifier(const std::string& filename, int num_labels);
		CostSpecifier(const std::vector<std::vector<double>>& cost_matrix, const std::vector<FeatureCostSpecifier>& feature_costs);

		void Initialize(const std::vector<FeatureCostSpecifier>& feature_costs);

		inline bool IsInitialized() const { return misclassification_costs.size() > 0; }
		double ComputeTotalTestCosts() const;
		double ComputeMaxMisclassificationCost() const;

		std::vector<std::vector<double>> misclassification_costs;
		std::vector<double> feature_costs;
		std::vector<double> discount_costs;
		std::vector<int> same_group;
		std::vector<int> same_binarized;
		double total_test_costs{ 0 };
		double max_misclassification_costs{ 0 };
		double max_instance_costs{ 0 };
	};

	class CostSensitive : public Classification {
	public:
		using SolType = double;
		using SolD2Type = double;
		using TestSolType = double;
		using BranchSolD2Type = double;

		static const bool total_order = true;
		static const bool custom_leaf = false;
		static const bool has_branching_costs = true;
		static const bool element_branching_costs = false;
		static const bool terminal_compute_context = true;
		static const bool terminal_zero_costs_true_label = false; // True iff the costs of assigning the true label in the terminal is zero
		static const int worst = INT32_MAX;
		static const int best = 0;

		CostSensitive(const ParameterHandler& parameters) : Classification(parameters) { UpdateParameters(parameters); }
		inline void UpdateParameters(const ParameterHandler& parameters) {
			cost_filename = parameters.GetStringParameter("cost-file");
		}
		inline void CopyTaskInfoFrom(const OptimizationTask* task) {
			UpdateCostSpecifier(static_cast<const CostSensitive*>(task)->cost_specifier);
		}
		void InformTrainData(const ADataView& train_data, const DataSummary& train_summary);
		void UpdateCostSpecifier(const CostSpecifier& cost_specifier) { this->cost_specifier = cost_specifier; }

		double GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const;
		inline double GetTestLeafCosts(const ADataView& data, const BranchContext& context, int label) const {
			return GetLeafCosts(data, context, label);
		}

		inline double GetBranchingCosts(const ADataView& data, const BranchContext& context, int feature) const { return data.Size() * GetBranchingCosts(context, feature); }
		inline double GetTestBranchingCosts(const ADataView& data, const BranchContext& context, int feature) const { return data.Size() * GetBranchingCosts(context, feature); }
		double GetBranchingCosts(const BranchContext& context, int feature) const;
		inline double ComputeD2BranchingCosts(const double& d2costs, int count) const { return d2costs * count; }

		void GetInstanceLeafD2Costs(const AInstance* instance, int org_label, int label, double& costs, int multiplier) const;
		inline void ComputeD2Costs(const double& d2costs, int count, double& costs) const { costs = d2costs; }
		inline bool IsD2ZeroCost(const double d2costs) const { return d2costs == 0; }
		inline double GetWorstPerLabel(int label) const { return cost_specifier.max_instance_costs; } // todo: differentiate per label

		inline double ComputeTrainScore(double test_value) const { return test_value; }
		double ComputeTrainTestScore(double test_value) const;
		double ComputeTestTestScore(double test_value) const;

		inline static bool CompareScore(double score1, double score2) { return score1 < score2; } // return true if score1 is better than score2

	private:	
		std::string cost_filename;
		CostSpecifier cost_specifier;
	};

}