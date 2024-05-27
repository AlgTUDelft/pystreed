#include "tasks/regression/regression.h"

namespace STreeD {

	Node<Regression> Regression::SolveLeafNode(const ADataView& data, const BranchContext& context) const {
		double ysq = 0, y = 0;
		for (const auto i : data.GetInstancesForLabel(0)) {
			auto instance = static_cast<const LInstance<double>*>(i);
			ysq += instance->GetLabel() * instance->GetLabel();
			y += instance->GetLabel();
		}
		double label = y / data.Size();
		double error = ysq - (y * y / data.Size());
		runtime_assert(error >= -1e-6);
		return Node<Regression>(label, error);
	}

	double Regression::GetLeafCosts(const ADataView& data, const BranchContext& context, double label) const {
		double ysq = 0;
		for (const auto i : data.GetInstancesForLabel(0)) {
			auto instance = static_cast<const LInstance<double>*>(i);
			ysq += (instance->GetLabel() - label) * (instance->GetLabel() - label);
		}
		return ysq;
	}

	void Regression::GetInstanceLeafD2Costs(const AInstance* instance, int org_label, int label, D2RegressionSol& costs, int multiplier) const {
		auto linstance = static_cast<const LInstance<double>*>(instance);
		costs.ys = multiplier * linstance->GetLabel();
		costs.yys = multiplier * linstance->GetLabel() * linstance->GetLabel();
	}

	void Regression::ComputeD2Costs(const D2RegressionSol& d2costs, int count, double& costs) const {
		if (count == 0) {
			costs = 0;
			return;
		}
		costs = d2costs.yys - (d2costs.ys * d2costs.ys / count); // MSE error
		costs = costs < 0 ? 0.0 : costs;
	}

	double Regression::GetLabel(const D2RegressionSol& costs, int count) const {
		if (count == 0) return 0;
		return costs.ys / count; // average of sum(y)
	}
}