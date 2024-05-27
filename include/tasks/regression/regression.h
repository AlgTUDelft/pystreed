#pragma once
#include "tasks/optimization_task.h"

namespace STreeD {
	
	struct D2RegressionSol {
		double ys{ 0 };
		double yys{ 0 };

		inline const D2RegressionSol& operator+=(const D2RegressionSol& v2) { ys += v2.ys; yys += v2.yys; return *this; }
		inline D2RegressionSol operator+(const D2RegressionSol& v2) const { return D2RegressionSol(*this) += v2; }
		inline const D2RegressionSol& operator-=(const D2RegressionSol& v2) { ys -= v2.ys; yys -= v2.yys; return *this; }
		inline D2RegressionSol operator-(const D2RegressionSol& v2) const { return D2RegressionSol(*this) -= v2; }
		inline bool operator==(const D2RegressionSol& v2) const { return std::abs(ys - v2.ys) < 1e-6 && std::abs(yys - v2.yys) < 1e-6; }
		inline bool operator!=(const D2RegressionSol& v2) const { return !(*this == v2); }
	};

	class Regression : public OptimizationTask {
	public:
		using SolType = double;
		using SolD2Type = D2RegressionSol;
		using TestSolType = double;
		using LabelType = double;
		using SolLabelType = double;

		static const bool total_order = true;
		static const bool custom_leaf = true;
		static constexpr  double worst = DBL_MAX;
		static constexpr  double best = 0;

		Regression(const ParameterHandler& parameters) {}

		Node<Regression> SolveLeafNode(const ADataView& data, const BranchContext& context) const;
		double GetLeafCosts(const ADataView& data, const BranchContext& context, double label) const;
		inline double GetTestLeafCosts(const ADataView& data, const BranchContext& context, double label) const {
			return GetLeafCosts(data, context, label);
		}
		double Classify(const AInstance*, double label) const { return label; }
		void GetInstanceLeafD2Costs(const AInstance* instance, int org_label, int label, D2RegressionSol& costs, int multiplier) const;
		void  ComputeD2Costs(const D2RegressionSol& d2costs, int count, double& costs) const;
		inline bool IsD2ZeroCost(const D2RegressionSol& d2costs) const { return d2costs.ys <= 1e-6 && d2costs.yys <= 1e-6 && d2costs.ys >= -1e-6 && d2costs.yys >= -1e-6; }
		double GetLabel(const D2RegressionSol& costs, int count) const;
		inline double GetWorstPerLabel(int label) const { return DBL_MAX; }

		inline double ComputeTrainScore(double train_value) const { return train_value / train_summary.size; }
		inline double ComputeTrainTestScore(double train_value) const { return train_value / train_summary.size; }
		inline double ComputeTestTestScore(double test_value) const { return test_value / test_summary.size; }
		inline static bool CompareScore(double score1, double score2) { return score1 < score2; } // return true if score1 is better than score2

	};

}