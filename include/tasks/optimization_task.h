#pragma once
#include "base.h"
#include "utils/parameter_handler.h"
#include "model/data.h"
#include "model/branch.h"
#include "model/node.h"
#include "model/container.h"
#include "solver/tuning.h"

namespace STreeD {

	// An empty context class
	class EmptyContext {

	};

	// The default branch context class
	class BranchContext {
	public:
		Branch& GetMutableBranch() { return branch; }
		const Branch& GetBranch() const { return branch; }
	private:
		Branch branch;
	};

	template <class OT>
	struct Tree;

	// The base optimization task class
	class OptimizationTask {
	public:
		using ET = ExtraData;				// The data type of the extra data for this optimization task
		using ContextType = BranchContext;	// The context type for this optimization task
		using BranchSolD2Type = int;		// The data type of the branching costs in the terminal solver
		static const bool use_terminal = true;			// True iff this OT uses the terminal solver
		static const bool element_additive = true;		// True iff this OT is element-additive, for using the similarity lower bound
		static const bool has_constraint = false;		// True iff this OT has constraints
		static const bool terminal_filter = false;		// True if you want to filter infeasible and UB dominated solutions in the terminal
		static const bool terminal_zero_costs_true_label = true; // True iff the costs of assigning the true label in the terminal is zero
		static const bool has_branching_costs = false;	// True iff this OT has branching costs
		static const bool constant_branching_costs = false; // True iff the branching costs are constant
		static const bool custom_lower_bound = false;	// True iff this task has a custom lower bound
		static const bool combine_custom_lb_purification = false; // True iff the custom lower bound of this task can be combined with the purification bound
		static const bool expensive_leaf = false;		// True iff this task has an expensive leaf node optimization function
		static const bool preprocess_data = false;		// True iff this task needs to preprocess the data after it is read from file
		static const bool preprocess_train_test_data = false;// True iff this task needs to preprocess the training data before training
		static const bool postprocess_tree = false;    // True iff this task needs to post-process
		static const bool custom_similarity_lb = false; // True iff this optimization task defines a custom sim.lb. bound
		static const bool terminal_compute_context = false; // True iff the optimization task requires the context to be computed in the terminal solver
		static const bool custom_get_label = false;			// Set to true if you want to compute the label customly, rather than using max. cost
		static const bool use_weights = false;				// Set to true if you want to counts to be based on the weights of isntances
		static constexpr int worst_label = INT32_MAX;		// Defines the default label, when no label is given.
		static constexpr int num_tune_phases = 1;			// Number of tuning phases in hypertuning
		static constexpr int minimum_difference = 0;		// The minimum difference between two solutions


		OptimizationTask() = default;

		// Inform the OT on the data that is used
		void InformTrainData(const ADataView& train_data, const DataSummary& train_summary);
		void InformTestData(const ADataView& test_data, const DataSummary& test_summary);
		
		// Return false if a feature is not available for branching
		bool MayBranchOnFeature(int feature) const { return true; }
		
		// Compare two score values. Default is maximization
		inline static bool CompareScore(double score1, double score2) { return score1 > score2; } // return true if score1 is better than score2

		// Update the context when branching
		void GetLeftContext(const ADataView& data, const BranchContext& context, int feature, BranchContext& left_context) const;
		void GetRightContext(const ADataView& data, const BranchContext& context, int feature, BranchContext& right_context) const;

		// Inform the OT on updated parameters (when hypertuning)
		inline void UpdateParameters(const ParameterHandler& parameters) { return; }
		inline void CopyTaskInfoFrom(const OptimizationTask* task) {}

		// Addition and subtraction functions
		inline static int Add(const int left, const int right) { return left + right; }
		inline static int TestAdd(const int left, const int right) { return left + right; }
		inline static void Add(const int left, const int right, int& out) { out = left + right; }
		inline static void Subtract(const int left, const int right, int& out) { out = std::max(0, left - right); }

		inline static double Add(const double left, const double right) { return left + right; }
		inline static double TestAdd(const double left, const double right) { return left + right; }
		inline static void Add(const double left, const double right, double& out) { out = left + right; }
		inline static void Subtract(const double left, const double right, double& out) { out = std::max(0.0, left - right); }

		// Print functions
		inline static std::string LabelToString(double sol_val) { return std::to_string(sol_val); }
		inline static std::string LabelToString(int sol_val) { return std::to_string(sol_val); }
		inline static std::string SolToString(double sol_val) { return std::to_string(sol_val); }
		inline static std::string SolToString(int sol_val) { return std::to_string(sol_val); }
		inline static std::string ScoreToString(double sol_val) { return std::to_string(sol_val); }
		inline static std::string ScoreToString(int sol_val) { return std::to_string(sol_val); }

		// Get the configurations for hypertuning
		static TuneRunConfiguration GetTuneRunConfiguration(const ParameterHandler& default_config, const ADataView& train_data, int phase);
		

	protected:
		DataSummary train_summary, test_summary;
	};

	class Classification : public OptimizationTask {
	public:
		using LabelType = int;
		using SolLabelType = int;

		Classification(const ParameterHandler& parameters) {}

		int Classify(const AInstance*, int label) const { return label; }
	};
}
