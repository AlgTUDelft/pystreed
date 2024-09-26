#pragma once
#include "tasks/optimization_task.h"
#include "tasks/regression/ckmeans/Ckmeans.1d.dp.h"

namespace STreeD {
	
	template <class OT>
	struct PairWorstCount;

	struct RegExtraData {
		double ysq{ 0 };
		int unique_feature_vector_id{ 0 };

		static RegExtraData ReadData(std::istringstream& iss, int num_labels) { return {}; }
	};

	struct D2CostComplexRegressionSol {
		double ys{ 0 };	 // The sum of y
		double yys{ 0 }; // The sum of y-squared
		int weight{ 0 }; // The sum of the weights

		inline const D2CostComplexRegressionSol& operator+=(const D2CostComplexRegressionSol& v2) { ys += v2.ys; yys += v2.yys; weight += v2.weight; return *this; }
		inline D2CostComplexRegressionSol operator+(const D2CostComplexRegressionSol& v2) const { return D2CostComplexRegressionSol(*this) += v2; }
		inline const D2CostComplexRegressionSol& operator-=(const D2CostComplexRegressionSol& v2) { ys -= v2.ys; yys -= v2.yys; weight -= v2.weight; return *this; }
		inline D2CostComplexRegressionSol operator-(const D2CostComplexRegressionSol& v2) const { return D2CostComplexRegressionSol(*this) -= v2; }
		inline bool operator==(const D2CostComplexRegressionSol& v2) const { return std::abs(ys - v2.ys) < 1e-6 && std::abs(yys - v2.yys) < 1e-6 && weight == v2.weight; }
		inline bool operator!=(const D2CostComplexRegressionSol& v2) const { return !(*this == v2); }
	};

	class CostComplexRegression : public OptimizationTask {
	public:
		using ET = RegExtraData;						// The extra data stores the y-squared value
		using SolType = double;							// The solution value is a Positive Real number
		using SolD2Type = D2CostComplexRegressionSol;	// For the terminal solver, the D2CostComplexRegressionSol is used to store the sums of y, y-squared and the weights
		using BranchSolD2Type = double;					// The data type of the branching costs in the terminal solver
		using TestSolType = double;						// The solution value (SSE) is a Real
		using LabelType = double;						// The label type (in the input): Regression, so Real
		using SolLabelType = double;					// The label type assigned to the leaf node, piecewise constant regression, so one Real number

		static const bool total_order = true;				// The solution values are totally ordered
		static const bool custom_leaf = true;				// A custom SolveLeafNode method is provided (to compute the mean of the data)
		static const bool custom_lower_bound = true;		// A custom lower bound is provided (kmeans)
		static const bool preprocess_data = true;			// The data is preprocessed (compute the y-squared values)
		static const bool preprocess_train_test_data = true;// The training data is preprocessed (sort on features, and group instances with same feature value)
		static const bool has_branching_costs = true;		// cost-complexity branching costs
		static const bool element_branching_costs = false;	// no branching costs for each element
		static const bool constant_branching_costs = true;  // The branching costs are constant (not depending on the size of the data)
		static const bool custom_similarity_lb = true;		// A custom similarity lower bound is provided
		static const bool use_weights = true;				// Set to true if you want to counts to be based on the weights of isntances
		static constexpr  double worst = DBL_MAX;			// The worst possible solution value
		static constexpr  double best = 0;					// The best possible solution value

		CostComplexRegression(const ParameterHandler& parameters)
			: cost_complexity_parameter(parameters.GetFloatParameter("cost-complexity")),
			use_kmeans(parameters.GetStringParameter("regression-bound") == "kmeans"),
			lower_bound_cache(parameters.GetIntegerParameter("max-depth") + 1) {
		}
		inline void UpdateParameters(const ParameterHandler& parameters) {
			cost_complexity_parameter = parameters.GetFloatParameter("cost-complexity");
			use_kmeans = parameters.GetStringParameter("regression-bound") == "kmeans";
			lower_bound_cache.resize(parameters.GetIntegerParameter("max-depth") + 1);
			minimum_leaf_node_size = std::max(1, int(parameters.GetIntegerParameter("min-leaf-node-size")));
		}
		void InformTrainData(const ADataView& train_data, const DataSummary& train_summary);
		void InformTestData(const ADataView& test_data, const DataSummary& test_summary);

		Node<CostComplexRegression> SolveLeafNode(const ADataView& data, const BranchContext& context) const;
		double GetLeafCosts(const ADataView& data, const BranchContext& context, double label) const;
		double GetTestLeafCosts(const ADataView& data, const BranchContext& context, double label) const;
		double Classify(const AInstance*, double label) const { return label; }
		void GetInstanceLeafD2Costs(const AInstance* instance, int org_label, int label, D2CostComplexRegressionSol& costs, int multiplier) const;

		double GetBranchingCosts(const ADataView& data, const BranchContext& context, int feature) const { return branching_cost; }
		double GetTestBranchingCosts(const ADataView& data, const BranchContext& context, int feature) const { return 0; }
		double GetBranchingCosts(const BranchContext& context, int feature) const { return branching_cost; }
		double ComputeD2BranchingCosts(const double& d2costs, int count) const { return d2costs; }

		void  ComputeD2Costs(const D2CostComplexRegressionSol& d2costs, int count, double& costs) const;
		inline bool IsD2ZeroCost(const D2CostComplexRegressionSol& d2costs) const { return d2costs.weight == 0; }
		double GetLabel(const D2CostComplexRegressionSol& costs, int count) const;
		//inline double GetWorstPerLabel(int label) const { return worst_distance_squared; }
		inline double GetWorstPerLabel(double label) const {
			double distMin = label - min;
			double distMax = max - label;

			return distMin > distMax ? distMin * distMin : distMax * distMax;
		}

		/*
		* Node only used as a container for the SolType, the label has no significance.
		*/
		Node<CostComplexRegression> ComputeLowerBound(const ADataView& data, const Branch& branch, int max_depth, int num_nodes);

		PairWorstCount<CostComplexRegression> ComputeSimilarityLowerBound(const ADataView& data_old, const ADataView& data_new) const;

		void PreprocessData(AData& data, bool train);
		void PreprocessTrainData(ADataView& train_data);
		void PreprocessTestData(ADataView& test_data) {}

		inline double ComputeTrainScore(double train_value) const { return train_value / total_training_weight; }
		inline double ComputeTrainTestScore(double train_value) const { return train_value / total_training_weight; }
		inline double ComputeTestTestScore(double test_value) const { return test_value / test_summary.size; }
		//inline double ComputeTestTestScore(double test_value) const { return 1 - test_value / test_total_variance; }
		inline static bool CompareScore(double score1, double score2) { return score1 < score2; } // return true if score1 is better than score2

		static TuneRunConfiguration GetTuneRunConfiguration(const ParameterHandler& default_config, const ADataView& data, int phase);

	private:
		// cache[i] is a hash table with branches of size i, mapping to a list where list[k - 1] is the lower bound for k max_nodes
		std::vector<std::unordered_map<const Branch, std::vector<Node<CostComplexRegression>>, BranchHashFunction, BranchEquality>> lower_bound_cache;
		// Reusable vectors for kmeans computation
		std::vector<WeightValuePair> weight_value_pairs;;
		std::vector<std::vector<double>> S;
		std::vector<std::vector<size_t>> J;
		double cost_complexity_parameter{ 0.01 };
		double branching_cost{ 0 };
		double worst_distance_squared{ 0 };
		double min{ 0 };
		double max{ 0 };
		double test_total_variance{ 1 };
		int minimum_leaf_node_size{ 1 };
		
		// extra data instances created in preprocessing
		AData data;
		
		// Reduce numerical instability by scaling
		double normalize_scale{ 1 };
		int total_training_weight{ 0 };

		bool use_kmeans{ false };
	};

}