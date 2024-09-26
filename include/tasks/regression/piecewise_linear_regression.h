#pragma once
#include "base.h"
#include "tasks/optimization_task.h"
#include "tasks/regression/ckmeans/Ckmeans.1d.dp.h"

namespace STreeD {
	
	template <class OT>
	struct PairWorstCount;

	template <class OT>
	struct Tree;

	struct PieceWiseLinearRegExtraData {
		std::vector<double> x;
		
		PieceWiseLinearRegExtraData() = default;
		PieceWiseLinearRegExtraData(const std::vector<double>& continuous_features) : x(continuous_features) {}
		
		int NumContFeatures() const { return int(x.size()); }

		static PieceWiseLinearRegExtraData ReadData(std::istringstream& iss, int num_extra_cols);
	};

	struct LinearModel {
		std::vector<double> b;
		double b0{ 0 };
		LinearModel() : b0(DBL_MAX) {}
		LinearModel(const std::vector<double>& b, double b0) : b(b), b0(b0) {}
		LinearModel(int num_cont_features, int b_ix, double b, double b0) : b(num_cont_features, 0.0), b0(b0) { this->b[b_ix] = b; }

		double Predict(const AInstance* instance) const;
		std::string ToString() const;

		bool operator==(const LinearModel& other) const;
		inline bool operator!=(const LinearModel& other) const { return !(*this == other); } 
	};
	

	class PieceWiseLinearRegression : public OptimizationTask {
	public:
		using ET = PieceWiseLinearRegExtraData;
		using SolType = double;				// Solution is a double (Regression)
		using TestSolType = double;			// Test solution is also a double
		using LabelType = double;			// The input label per instance is a double
		using SolLabelType = LinearModel;	// the output label (on each leaf node) is a linear model

		static const bool total_order = true;	// Solutions are totally ordered by SSE
		static const bool custom_leaf = true;	// This OT defines its own leaf node optimization function
		static const bool expensive_leaf = true; // This OT has an expensive leaf node optimization function
		static const bool custom_lower_bound = false; // This OT does not define a custom LB
		static const bool preprocess_data = true;	  // This OT preprocesses the data (normalization)
		static const bool preprocess_train_test_data = true;	// This OT preprocesses the train and test data
		static const bool postprocess_tree = true;    // This task post processes the tree (reverse normalization)
		static const bool has_branching_costs = true; // This task has branching costs (complexity costs)
		static const bool element_branching_costs = false; // These costs are not per instance
		static const bool constant_branching_costs = true; // These costs are constant
		static const bool custom_similarity_lb = true;     // This task defines a custom sim. LB
		static const bool use_terminal = false;			   // This task does not use the terminal solver
		static constexpr double worst = DBL_MAX;	// The worst solution is Inf.
		static constexpr double best = 0;			// The best solutino is zero
		static const LinearModel worst_label;
		static constexpr int num_tune_phases = 3;	// Tune in three phases: 1) the lasso and ridge penalties, 2) the complexity cost, 3) again the lasso and ridge penalties

		PieceWiseLinearRegression(const ParameterHandler& parameters) {
			UpdateParameters(parameters);
		}
		inline void UpdateParameters(const ParameterHandler& parameters) {
			cost_complexity_parameter = parameters.GetFloatParameter("cost-complexity");
			lasso_penalty = parameters.GetFloatParameter("lasso-penalty");
			ridge_penalty = parameters.GetFloatParameter("ridge-penalty");
			min_leaf_node_size = int(parameters.GetIntegerParameter("min-leaf-node-size"));
			int cf = int(parameters.GetIntegerParameter("num-extra-cols"));
			if (cf > min_leaf_node_size) {
				std::cout << "Piecewise linear regression requires at least the number of continuous features as the minimum leaf node size." << std::endl;
				std::exit(1);
			}
		}
		void InformTrainData(const ADataView& train_data, const DataSummary& train_summary);
		void InformTestData(const ADataView& test_data, const DataSummary& test_summary);

		Node<PieceWiseLinearRegression> SolveLeafNode(const ADataView& data, const BranchContext& context) const;
		double GetLeafCosts(const ADataView& data, const BranchContext& context, const LinearModel& label) const;
		double GetTestLeafCosts(const ADataView& data, const BranchContext& context, const LinearModel& label) const;
		double Classify(const AInstance* instance, const LinearModel& label) const { return label.Predict(instance); }
		double GetBranchingCosts(const ADataView& data, const BranchContext& context, int feature) const { return branching_cost; }
		double GetTestBranchingCosts(const ADataView& data, const BranchContext& context, int feature) const { return 0; }
		double GetBranchingCosts(const BranchContext& context, int feature) const { return branching_cost; }

		
		inline double GetWorstPerLabel(double label) const {
			double max_diff = (max_lb - std::abs(label));
			return max_diff * max_diff;
		}

		PairWorstCount<PieceWiseLinearRegression> ComputeSimilarityLowerBound(const ADataView& data_old, const ADataView& data_new) const;

		void PreprocessData(AData& data, bool train);
		void PreprocessTrainData(ADataView& train_data);
		void PreprocessTestData(ADataView& test_data) {}
		void PostProcessTree(std::shared_ptr<Tree<PieceWiseLinearRegression>> tree);

		inline double ComputeTrainScore(double train_value) const { return train_value / train_summary.size; }
		inline double ComputeTrainTestScore(double train_value) const { return train_value / train_summary.size; }
		
		inline double ComputeTestTestScore(double test_value) const { return test_value / test_summary.size; } // Report MSE

		//inline double ComputeTestTestScore(double test_value) const { return 1.0 - test_value / test_total_variance; } //Report R^2

		inline static bool CompareScore(double score1, double score2) { return score1 < score2; } // return true if score1 is better than score2

		inline static std::string LabelToString(const LinearModel& label) { return label.ToString(); }
		inline static std::string SolToString(double sol_val) { return std::to_string(sol_val); }

		static TuneRunConfiguration GetTuneRunConfiguration(const ParameterHandler& default_config, const ADataView& data, int phase);

	private:
		std::pair<LinearModel, double> SolveGLMNet(const ADataView& data) const;

		// lambda penalty factor (percentage of max-lambda)
		double lasso_penalty{ 0 };
		double ridge_penalty{ 0 };

		double cost_complexity_parameter{ 0.01 };
		double branching_cost{ 0 };
		
		// Similarity bound values
		double max_lb{ 0 };
		
		// Reduce numerical instability by scaling
		double cost_complexity_norm_coef{ 1 };
		double y_mu{ 0 }, y_std{ 1 };
		std::vector<double> g_cf_mu, g_cf_std;
		int min_leaf_node_size{ 1 };
		double test_total_variance{ 1 };

		mutable Branch last_branch;
		mutable Node<PieceWiseLinearRegression> last_solution;

		// Variables for GLMNet
		mutable std::vector<std::vector<double>> X, x1x2;
		mutable std::vector<double> cf_y, cf_ysq, cf_mu, cf_std,
			b, xy, x;// xx, y;
		mutable std::vector<int> redundant_feature, is_non_zero;

		// extra data instances created in preprocessing
		AData data;
	};

}