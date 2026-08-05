#pragma once
#include "tasks/optimization_task.h"

namespace STreeD {
		
	enum class AccuracyObjective {
		misclassification_score,
		gini_index,
		sqrt_gini,
		minimum_error,
		entropy,
		mdl_quinlan,
		pessimistic_binomial,
		mdl_mehta,
		bayes,
		mloss,
		lloss,
	};

	enum class AccuracyTuningMethod {
		depth,
		tree_size,
		cc,
		wcc,
		min_support,
		smoothing
	};

	// Optimization task that computes an optimal classification tree for a varied set of
	// Leaf node objectives and tuning methods. Only binary labels are supported!
	class AccuracyFlex : public Classification {
	public:
		using SolType = double;			// The data type of the solution
		using SolD2Type = int;			// The data type of the solution in the terminal solver
		using BranchSolD2Type = double;	// The data type of the branching costs in the terminal solver
		using TestSolType = int;		// The data type of the solution that is used for evaluation

		static const bool total_order = true;		// True iff the OT is totally ordered 
		static const bool custom_leaf = true;		// Set to true if you want to implement a custom leaf function (for optimization)
		static const bool element_additive = false; // Do not use the similarity lower bound
		static const bool has_branching_costs = true;	// Has branching costs (when using cost-complexity tuning
		static const bool element_branching_costs = false; // Branching costs do not depend on elements
		static const bool constant_branching_costs = false; // Branching costs are not constant (for weighted cost complexity tuning)
		static constexpr int worst = INT32_MAX;		// An UB for the worst solution value possible
		static constexpr int best = 0;				// A LB for the best solution value possible

		AccuracyFlex(const ParameterHandler& parameters) : Classification(parameters) { UpdateParameters(parameters); }

		void InformTrainData(const ADataView& train_data, const DataSummary& train_summary);

		void UpdateParameters(const ParameterHandler& parameters);

		// Find the optimal leaf node solution for the data in the given context
		Node<AccuracyFlex> SolveLeafNode(const ADataView& data, const BranchContext& context) const;

		// Compute the leaf costs for the data in the context when assigning label
		double GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const;

		// Compute the test leaf costs for the data in the context when assigning label
		int GetTestLeafCosts(const ADataView& data, const BranchContext& context, int label) const;

		// Get the branching costs
		double GetBranchingCosts(const ADataView& data, const BranchContext& context, int feature) const { 
			return cost_complexity_parameter * (weighted_cost_complexity ? data.Size() : train_summary.size); }
		
		// Return zero branching costs for the test
		int GetTestBranchingCosts(const ADataView& data, const BranchContext& context, int feature) const { return 0; }
		
		// Get the branching costs
		double GetBranchingCosts(const BranchContext& context, int feature) const {
			return cost_complexity_parameter * (weighted_cost_complexity ? 1 : train_summary.size);
		}

		// Turn the D2 branching costs into normal branching costs
		double ComputeD2BranchingCosts(const double& d2costs, int count) const { return d2costs * (weighted_cost_complexity ? count : 1); }

		// Compute the leaf costs for an instance given a assigned label
		inline void GetInstanceLeafD2Costs(const AInstance* instance, int org_label, int label, int& costs, int multiplier) const { costs = multiplier * ((org_label == label) ? 0 : 1); }

		// Compute the solution value from a terminal solution value
		void ComputeD2Costs(const int& d2costs, int count, int label, int depth, double& costs) const;

		// Return true if the terminal solution value is zero
		inline bool IsD2ZeroCost(const int d2costs) const { return d2costs == 0; }

		// Get a bound on the worst contribution to the objective of a single instance with label
		inline int GetWorstPerLabel(int label) const { return 1; }

		// Compute the train score from the training solution value
		inline double ComputeTrainScore(double test_value) const { return ((double)(train_summary.size - test_value)) / ((double)train_summary.size); }

		// Compute the test score on the training data from the test solution value
		inline double ComputeTrainTestScore(int test_value) const { return ((double)(train_summary.size - test_value)) / ((double)train_summary.size); }

		// Compute the test score on the test data from the test solution value
		inline double ComputeTestTestScore(int test_value) const { return ((double)(test_summary.size - test_value)) / ((double)test_summary.size); }

		// Number of tuning phases in hypertuning 
		int GetNumberOfTunePhases() const { return 1; }//objective_type == AccuracyObjective::node_hom_class_recov ? 2 : 1;

		static TuneRunConfiguration GetTuneRunConfiguration(const ParameterHandler& default_config, const ADataView& data, int phase);

	private:
		int GetError(const ADataView& data, int label) const;
		double GetMisclassificationScore(int n, int e, int label) const;
		double GetGiniScore(int n, int e) const;
		double GetSqrtGiniScore(int n, int e) const;
		double GetMinError(int n, int e) const;
		double GetEntropy(int n, int e) const;
		double GetMDLQuinlan(int n, int e) const;
		double GetMDLMehta(int n, int e) const;
		double GetObjectiveError(int n, int e, int label, int depth) const;
		double GetPessimisticErrorIncrease(int n, int e) const;
		double GetBayesError(int n, int e, int depth) const;
		double GetMLossError(int n, int e) const;
		double GetLLossError(int n, int e) const;

		std::vector<double> lg_ncrs;
		double CF{ 0 };
		double coef{ 0 };
		double cost_complexity_parameter{ 0.00 };
		bool weighted_cost_complexity{ false };
		int smoothing{ 0 };
		AccuracyObjective objective_type{ AccuracyObjective::misclassification_score };
		AccuracyTuningMethod tuning_method{ AccuracyTuningMethod::tree_size };
	};
}