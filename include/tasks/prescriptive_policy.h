#pragma once
#include "tasks/optimization_task.h"

namespace STreeD {

	class PPGData {
	public:
		static PPGData ReadData(std::istringstream& iss, int num_labels);
		PPGData() = default;
		PPGData(int k, double y, double mu,
			std::vector<double>& yhat, int k_opt,
			std::vector<double>& cf_y) 
			: k(k), y(y), mu(mu), yhat(yhat), k_opt(k_opt), cf_y(cf_y) { ProcessData(); }
		PPGData(int k, double y, double mu, std::vector<double>& yhat) 
			: k(k), y(y), mu(mu), yhat(yhat), k_opt(0), cf_y(std::vector<double>(yhat.size(), 0)) { ProcessData(); }
		
		// For training
		int k{ 0 };						    // Historic treatment k 
		double y{ 0 };						// Historic outcome y
		double mu{ 0 };						// Propensity score mu
		std::vector<double> yhat;			// Regress & Compare prediction yhat (for every label)
		// For testing
		int k_opt{ 0 };						// The optimal treatment k_opt
		std::vector<double> cf_y;			// The (possibly counterfactual) outcome cf_y (for every label)
		// Derived values
		std::vector<double> dm_scores;		// direct method score (for every label)
		std::vector<double> ipw_scores;		// inverse propensity weight score (for every label)
		std::vector<double>	dr_scores;		// double robust score (for every label)
	private:
		void ProcessData();
	};

	class PrescriptivePolicy : public Classification {
	public:
		using SolType = double;
		using SolD2Type = double;
		using TestSolType = int;
		using ET = PPGData;

		static const bool total_order = true;
		static const bool custom_leaf = false;
		static const bool terminal_zero_costs_true_label = false; // True iff the costs of assigning the true label in the terminal is zero
		static const int worst = INT32_MAX;
		static const int best = 0;

		PrescriptivePolicy(const ParameterHandler& parameters);
		void InformTrainData(const ADataView& train_data, const DataSummary& train_summary);

		double GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const;
		int GetTestLeafCosts(const ADataView& data, const BranchContext& context, int label) const;
		void GetInstanceLeafD2Costs(const AInstance* instance, int org_label, int label, double& costs, int multiplier) const;
		void ComputeD2Costs(const double& d2costs, int count, double& costs) const { costs = d2costs; }
		inline bool IsD2ZeroCost(const double d2costs) const { return std::abs(d2costs) < 1e-6; }
		inline double GetWorstPerLabel(int label) const { return worst_per_label[label]; }

		// For training, return costs
		double ComputeTrainScore(double test_value) const { return test_value; }
		// For tests, return out-of-sample correct treatment
		double ComputeTrainTestScore(int test_value) const { return ((double)(train_summary.size - test_value)) / ((double)train_summary.size); }
		double ComputeTestTestScore(int test_value) const { return ((double)(test_summary.size - test_value)) / ((double)test_summary.size); }
	private:
		bool use_dm{ false };
		bool use_ipw{ false };
		bool use_dr{ false };
		double min_dm{ 0 };
		double min_ipw{ 0 };
		double min_dr{ 0 };
		std::vector<double> worst_per_label;
		std::vector<std::vector<double>> cost_per_training_instance;
	};

}