#pragma once
#include "tasks/optimization_task.h"

namespace STreeD {

	class CostComplexAccuracy : public Classification {
	public:
		using SolType = int;
		using SolD2Type = int;
		using TestSolType = int;

		static const bool total_order = true;
		static const bool custom_leaf = false;
		static const bool element_additive = true;
		static const bool has_branching_costs = true;
		static const bool element_branching_costs = false;
		static const bool constant_branching_costs = true;
		static constexpr int worst = INT32_MAX;
		static constexpr int best = 0;

		CostComplexAccuracy(const ParameterHandler& parameters) 
			: Classification(parameters), cost_complexity_parameter(parameters.GetFloatParameter("cost-complexity")) {}

		inline void UpdateParameters(const ParameterHandler& parameters) {
			cost_complexity_parameter = parameters.GetFloatParameter("cost-complexity");
		}

		int GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const;
		inline int GetTestLeafCosts(const ADataView& data, const BranchContext& context, int label) const {
			return int(GetLeafCosts(data, context, label));
		}

		int GetBranchingCosts(const ADataView& data, const BranchContext& context, int feature) const { return int(cost_complexity_parameter * train_summary.size); }
		int GetTestBranchingCosts(const ADataView& data, const BranchContext& context, int feature) const { return 0; }
		int GetBranchingCosts(const BranchContext& context, int feature) const { 
			double val = cost_complexity_parameter * train_summary.size;
			int int_val = int(val);
			runtime_assert(int_val >= -1e-6);
			return int_val; }
		int ComputeD2BranchingCosts(const int& d2costs, int count) const { return d2costs; }

		inline void GetInstanceLeafD2Costs(const AInstance* instance, int org_label, int label, int& costs, int multiplier) const { costs = multiplier * ((org_label == label) ? 0 : 1); }
		void ComputeD2Costs(const int& d2costs, int count, int& costs) const { costs = d2costs; }
		inline bool IsD2ZeroCost(const int d2costs) const { return d2costs == 0; }
		inline int GetWorstPerLabel(int label) const { return 1; }

		inline double ComputeTrainTestScore(int test_value) const { return ((double)(train_summary.size - test_value)) / ((double)train_summary.size); }
		inline double ComputeTestTestScore(int test_value) const { return ((double)(test_summary.size - test_value)) / ((double)test_summary.size); }

		static TuneRunConfiguration GetTuneRunConfiguration(const ParameterHandler& default_config, const ADataView& data, int phase);

	private:
		double cost_complexity_parameter{ 0.01 };
	};

}