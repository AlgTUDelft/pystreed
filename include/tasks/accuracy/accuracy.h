#pragma once
#include "tasks/optimization_task.h"

namespace STreeD {

	class Accuracy : public Classification {
	public:
		using SolType = int;			// The data type of the solution
		using SolD2Type = int;			// The data type of the solution in the terminal solver
		using TestSolType = int;		// The data type of the solution that is used for evaluation

		static const bool total_order = true;		// True iff the OT is totally ordered 
		static const bool custom_leaf = false;		// Set to true if you want to implement a custom leaf function (for optimization)
		static constexpr int worst = INT32_MAX;		// An UB for the worst solution value possible
		static constexpr int best = 0;				// A LB for the best solution value possible
		static constexpr int minimum_difference = 1;// The minimum difference between two solutions

		Accuracy(const ParameterHandler& parameters) : Classification(parameters) {}

		// Compute the leaf costs for the data in the context when assigning label
		int GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const;
		
		// Compute the test leaf costs for the data in the context when assigning label
		inline int GetTestLeafCosts(const ADataView& data, const BranchContext& context, int label) const {
			return GetLeafCosts(data, context, label);
		}

		// Compute the leaf costs for an instance given a assigned label
		inline void GetInstanceLeafD2Costs(const AInstance* instance, int org_label, int label, int& costs, int multiplier) const { costs = multiplier * ((org_label == label) ? 0 : 1); }
		
		// Compute the solution value from a terminal solution value
		void ComputeD2Costs(const int& d2costs, int count, int& costs) const { costs = d2costs; }
		
		// Return true if the terminal solution value is zero
		inline bool IsD2ZeroCost(const int d2costs) const { return d2costs == 0; }
		
		// Get a bound on the worst contribution to the objective of a single instance with label
		inline int GetWorstPerLabel(int label) const { return 1; }
		
		// Compute the train score from the training solution value
		inline double ComputeTrainScore(int test_value) const { return ((double)(train_summary.size - test_value)) / ((double)train_summary.size); }
		
		// Compute the test score on the training data from the test solution value
		inline double ComputeTrainTestScore(int test_value) const { return ((double)(train_summary.size - test_value)) / ((double)train_summary.size); }
		
		// Compute the test score on the test data from the test solution value
		inline double ComputeTestTestScore(int test_value) const { return ((double)(test_summary.size - test_value)) / ((double)test_summary.size); }
	};

}