#pragma once
#include "tasks/optimization_task.h"
#include "tasks/regression/piecewise_linear_regression.h"

namespace STreeD {
	
	template <class OT>
	struct PairWorstCount;

	struct SimpleLinRegExtraData : PieceWiseLinearRegExtraData {
		SimpleLinRegExtraData() = default;
		SimpleLinRegExtraData(int num_features) : PieceWiseLinearRegExtraData(x), xsq(num_features, 0.0), yx(num_features, 0.0) {}
		SimpleLinRegExtraData(const std::vector<double>& x);
		
		double ysq{ 0 };
		std::vector<double> xsq, yx;

		static SimpleLinRegExtraData ReadData(std::istringstream& iss, int num_labels);
	};

	struct D2SimpleLinRegSol {
		double ys{ 0 };
		double yys{ 0 };
		int weight{ 0 };
		std::vector<double> xsq, yx, xs;

		const D2SimpleLinRegSol& operator+=(const D2SimpleLinRegSol& v2);
		inline D2SimpleLinRegSol operator+(const D2SimpleLinRegSol& v2) const { return D2SimpleLinRegSol(*this) += v2; }
		const D2SimpleLinRegSol& operator-=(const D2SimpleLinRegSol& v2);
		inline D2SimpleLinRegSol operator-(const D2SimpleLinRegSol& v2) const { return D2SimpleLinRegSol(*this) -= v2; }
		bool operator==(const D2SimpleLinRegSol& v2) const;
		inline bool operator!=(const D2SimpleLinRegSol& v2) const { return !(*this == v2); }
	};

	class SimpleLinearRegression : public OptimizationTask {
	public:
		using ET = SimpleLinRegExtraData;
		using SolType = double;
		using SolD2Type = D2SimpleLinRegSol;
		using BranchSolD2Type = double;
		using TestSolType = double;
		using LabelType = double;
		using SolLabelType = LinearModel;

		static const bool total_order = true;
		static const bool custom_leaf = true;
		static const bool expensive_leaf = true; // This OT has an expensive leaf node optimization function
		static const bool preprocess_data = true;
		static const bool preprocess_train_test_data = true;
		static const bool has_branching_costs = true;
		static const bool element_branching_costs = false;
		static const bool constant_branching_costs = true;
		static const bool custom_similarity_lb = true; 
		static const bool element_additive = true; 
		static const bool use_terminal = true; 
		static const bool terminal_zero_costs_true_label = false; // True iff the costs of assigning the true label in the terminal is zero
		static constexpr double worst = DBL_MAX;
		static constexpr double best = 0;
		static constexpr bool use_weights = true;
		static const LinearModel worst_label;
		static constexpr int num_tune_phases = 3;

		SimpleLinearRegression(const ParameterHandler& parameters) {
			UpdateParameters(parameters);
		}
		inline void UpdateParameters(const ParameterHandler& parameters) {
			cost_complexity_parameter = parameters.GetFloatParameter("cost-complexity");
			ridge_penalty = parameters.GetFloatParameter("lasso-penalty");
			minimum_leaf_node_size = std::max(1, int(parameters.GetIntegerParameter("min-leaf-node-size")));
		}
		void InformTrainData(const ADataView& train_data, const DataSummary& train_summary);
		void InformTestData(const ADataView& test_data, const DataSummary& test_summary);

		Node<SimpleLinearRegression> SolveLeafNode(const ADataView& data, const BranchContext& context) const;
		double GetLeafCosts(const ADataView& data, const BranchContext& context, const LinearModel& model) const;
		double GetTestLeafCosts(const ADataView& data, const BranchContext& context, const LinearModel& model) const;
		double Classify(const AInstance* instance, const LinearModel& model) const { return model.Predict(instance); }
		void GetInstanceLeafD2Costs(const AInstance* instance, int org_label, int label, D2SimpleLinRegSol& costs, int multiplier) const;

		double GetBranchingCosts(const ADataView& data, const BranchContext& context, int feature) const { return branching_cost; }
		double GetTestBranchingCosts(const ADataView& data, const BranchContext& context, int feature) const { return 0; }
		double GetBranchingCosts(const BranchContext& context, int feature) const { return branching_cost; }
		double ComputeD2BranchingCosts(const double& d2costs, int count) const { return d2costs; }

		void  ComputeD2Costs(const D2SimpleLinRegSol& d2costs, int count, double& costs) const;
		inline bool IsD2ZeroCost(const D2SimpleLinRegSol& d2costs) const { return d2costs.weight == 0; }
		LinearModel GetLabel(const D2SimpleLinRegSol& costs, int count) const;
		
		inline double GetWorstPerLabel(double label) const {
			double distMin = label - min;
			double distMax = max - label;

			return distMin > distMax ? distMin * distMin : distMax * distMax;
		}

		PairWorstCount<SimpleLinearRegression> ComputeSimilarityLowerBound(const ADataView& data_old, const ADataView& data_new) const;

		void PreprocessData(AData& data, bool train);
		void PreprocessTrainData(ADataView& train_data);
		void PreprocessTestData(ADataView& test_data) {}

		inline double ComputeTrainScore(double train_value) const { return train_value / total_training_weight; }
		inline double ComputeTrainTestScore(double train_value) const { return train_value / total_training_weight; }
		inline double ComputeTestTestScore(double test_value) const { return test_value / test_summary.size; }
		//inline double ComputeTestTestScore(double test_value) const { return 1 - test_value / test_total_variance; }
		inline static bool CompareScore(double score1, double score2) { return score1 < score2; } // return true if score1 is better than score2

		inline static std::string LabelToString(const LinearModel& label) { return label.ToString(); }
		inline static std::string SolToString(double sol_val) { return std::to_string(sol_val); }

		static TuneRunConfiguration GetTuneRunConfiguration(const ParameterHandler& default_config, const ADataView& data, int phase);

	private:
		double cost_complexity_parameter{ 0.01 };
		double ridge_penalty{ 0.001 };
		double branching_cost{ 0 };
		double worst_distance_squared{ 0 };
		double min{ 0 };
		double max{ 0 };
		double test_total_variance{ 1 };
		std::vector<double> feature_variance;

		// extra data instances created in preprocessing
		AData data;

		int total_training_weight{ 0 };
		int num_cont_features{ 0 };
		int minimum_leaf_node_size{ 1 };

		// Store last solution in solve leaf node
		mutable Branch last_branch;
		mutable Node<SimpleLinearRegression> last_solution;
	};

}