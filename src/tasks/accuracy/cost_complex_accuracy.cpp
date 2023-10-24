#include "tasks/accuracy/cost_complex_accuracy.h"

namespace STreeD {

	int CostComplexAccuracy::GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const { // Replace by custom function later
		int error = 0;
		for (int k = 0; k < data.NumLabels(); k++) {
			if (k == label) continue;
			error += data.NumInstancesForLabel(k);
		}
		return error;
	}

	TuneRunConfiguration CostComplexAccuracy::GetTuneRunConfiguration(const ParameterHandler& default_config, const ADataView& data, int tune_phase) {
		TuneRunConfiguration config;

		int max_nodes = int(default_config.GetIntegerParameter("max-num-nodes"));
		int max_d = int(default_config.GetIntegerParameter("max-depth"));

		double base_alpha = 1.0 / data.Size();
		std::vector<double> alphas;
		for (int a = 1; a < 10; a++)
			alphas.push_back(base_alpha * a);
		for (int a = 10; a <= 100; a += 10)
			alphas.push_back(base_alpha * a);
		for(double alpha = 100*base_alpha; alpha < 0.01; alpha += 0.001)
			alphas.push_back(alpha);
		for (auto a: alphas) {
			if (a > 0.1) continue;
			ParameterHandler params = default_config;
			params.SetFloatParameter("cost-complexity", a);
			config.AddConfiguration(params, "a = " + std::to_string(a));
		}
		config.reset_solver = true;
		config.skip_when_max_tree = true;
		return config;
	}

}