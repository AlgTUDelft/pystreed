#include "tasks/accuracy/accuracy_flex.h"

namespace STreeD {

	AccuracyTuningMethod GetAccuracyTuningMethod(const ParameterHandler& parameters) {
		std::string tuning_str = parameters.GetStringParameter("tune-method");
		if (tuning_str == "tree-size") {
			return AccuracyTuningMethod::tree_size;
		} else if (tuning_str == "depth") {
			return AccuracyTuningMethod::depth;
		} else if (tuning_str == "cost-complexity") {
				return AccuracyTuningMethod::cc;
		} else if (tuning_str == "weighted-cost-complexity") {
			return AccuracyTuningMethod::wcc;
		} else if (tuning_str == "min-leaf-node-size") {
			return AccuracyTuningMethod::min_support;
		} else if (tuning_str == "smoothing") {
			return AccuracyTuningMethod::smoothing;
		} else {
			std::cout << "Provide a valid value for the tune-method. '" << tuning_str << "' is invalid." << std::endl;
			exit(1);
		}
	}

	AccuracyObjective GetAccuracyObjective(const ParameterHandler& parameters) {
		std::string objective_str = parameters.GetStringParameter("accuracy-objective");
		if (objective_str == "misclassification-score") {
			return AccuracyObjective::misclassification_score;
		} else if (objective_str == "gini-index") {
			return AccuracyObjective::gini_index;
		} else if (objective_str == "sqrt-gini") {
			return AccuracyObjective::sqrt_gini;
		} else if (objective_str == "min-error") {
			return AccuracyObjective::minimum_error;
		} else if (objective_str == "entropy") {
			return  AccuracyObjective::entropy;
		} else if (objective_str == "mdl-quinlan") {
			return  AccuracyObjective::mdl_quinlan;
		} else if (objective_str == "mdl-mehta") {
			return  AccuracyObjective::mdl_mehta;
		} else if (objective_str == "pessimistic-binomial") {
			return  AccuracyObjective::pessimistic_binomial;
		} else if (objective_str == "bayes") {
			return AccuracyObjective::bayes;
		} else if (objective_str == "m-loss") {
			return  AccuracyObjective::mloss;
		} else if (objective_str == "l-loss") {
			return  AccuracyObjective::lloss;
		} else {
			std::cout << "Provide a valid value for the accuracy-objective. '" << objective_str << "' is invalid." << std::endl;
			exit(1);
		}
	}

	void AccuracyFlex::UpdateParameters(const ParameterHandler& parameters) {
		// Determine the objective type
		objective_type = GetAccuracyObjective(parameters);
		tuning_method = GetAccuracyTuningMethod(parameters);

		// Update the cost-complexity parameter
		
		weighted_cost_complexity = tuning_method == AccuracyTuningMethod::wcc;
			//(tune_method == "weighted-cost-complexity") || (tune_method == "prune-weighted-cost-complexity");
		if (tuning_method == AccuracyTuningMethod::cc || tuning_method == AccuracyTuningMethod::wcc) {
			cost_complexity_parameter = parameters.GetFloatParameter("cost-complexity");
		} else {
			cost_complexity_parameter = 0.0;
		}

		// Update the class recovery factor parameter
		smoothing = int(parameters.GetIntegerParameter("smoothing"));

		// Determine the pessimistic coefficient (from C4.5)
		CF = parameters.GetFloatParameter("confidence-coefficient");
		double val[] = { 0,  0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.40, 1.00 },
			dev[] = { 4.0,  3.09,  2.58,  2.33, 1.65, 1.28, 0.84, 0.25, 0.00 };
		int i = 0;
		while (CF > val[i]) i++;
		if (i == 0) {
			std::cout << "Provide a valid value for the confidence-coefficient, higher than 0 and below 1.0, instead of " << CF << std::endl;
			exit(1);
			return;
		}
		coef = dev[i - 1] +
			(dev[i] - dev[i - 1]) * (CF - val[i - 1]) / (val[i] - val[i - 1]);
		coef = coef * coef;
	}

	void AccuracyFlex::InformTrainData(const ADataView& train_data, const DataSummary& train_summary) {
		OptimizationTask::InformTrainData(train_data, train_summary);
		if (train_data.NumLabels() > 2
			&& objective_type != AccuracyObjective::misclassification_score
			&& objective_type != AccuracyObjective::pessimistic_binomial) {
			std::cout << "This optimization task can only be used for binary classification." << std::endl;
			exit(1);
		}
		// Precompute log (ncr (n,k)) for MDL
		lg_ncrs.resize(train_data.Size() + 1);
		lg_ncrs[0] = 0;
		for (int i = 1; i <= train_data.Size(); i++) {
			lg_ncrs[i] = lg_ncrs[i - 1] + std::log2(i);
		}
	}

	// Get pessimistic increase in error from the binomial distribution
	// From Quinlan, C4.5: Programs for Machine Learning (1993)
	double AccuracyFlex::GetPessimisticErrorIncrease(int N, int e) const {
		if (e < 1E-6) {
			return N * (1 - exp(log(CF) / N));
		} else if (e < 0.9999) {
			double val0 = N * (1 - exp(log(CF) / N));
			return val0 + e * (GetPessimisticErrorIncrease(N, 1) - val0);
		} else if (e + 0.5 >= N) {
			return 0.67 * (N - e);
		} else {
			double pr = (e + 0.5 + coef / 2
				+ sqrt(coef * ((e + 0.5) * (1 - (e + 0.5) / N) + coef / 4.0)))
				/ (N + coef);
			return (N * pr - e);
		}
	}

	int AccuracyFlex::GetError(const ADataView& data, int label) const {
		int error = 0;
		for (int k = 0; k < data.NumLabels(); k++) {
			if (k == label) continue;
			error += data.NumInstancesForLabel(k);
		}
		return error;
	}

	double AccuracyFlex::GetMisclassificationScore(int n, int e, int label) const {
		return e;
	}

	double AccuracyFlex::GetGiniScore(int n, int e) const {
		double p = ((double) e )/ n;
		double error = p * p + (1 - p) * (1 - p);
		return n * (1-error);
	}

	double AccuracyFlex::GetSqrtGiniScore(int n, int e) const {
		double p = ((double)e) / n;
		double error = p * p + (1 - p) * (1 - p);
		return n * std::sqrt(1 - error);
	}

	double AccuracyFlex::GetMinError(int n, int e) const {
		return ((double) n * (e + 1)) / ((double) (n + 2));
	}

	double AccuracyFlex::GetEntropy(int n, int e) const {
		if (e == 0 || e == n) return 0;
		double p = ((double)e) / n;
		double error = p * std::log2(p) + (1 - p) * std::log2(1 - p);
		return -n * error;
	}

	double AccuracyFlex::GetMDLQuinlan(int n, int e) const {
		runtime_assert(n <= lg_ncrs.size());
		runtime_assert(e <= (n + 1) / 2);
		runtime_assert(lg_ncrs[n] - lg_ncrs[e] - lg_ncrs[n - e] >= -1e-5);
		return std::log2((n + 1) / 2 + 1) + lg_ncrs[n] - lg_ncrs[e] - lg_ncrs[n - e];
	}

	double AccuracyFlex::GetMDLMehta(int n, int e) const {
		double error = 0;
		if (e != 0) {
			double n_div_e = ((double)n) / ((double)e);
			error += e * std::log(n_div_e);
		}
		if (e != n) {
			double n_div_n_min_e = ((double)n) / ((double)(n - e));
			error += (n - e) * std::log(n_div_n_min_e);
		}
		double n_div_2 = ((double)n / 2.0);
		const double gamma_1 = 1.0;
		const double pi = 3.14159265358979323846;
		error += 0.5 * std::log(n_div_2) + std::log(pi / gamma_1);
		return error;
	}

	double logBeta(double alpha, double beta) {
		return std::lgamma(alpha) + std::lgamma(beta) - std::lgamma(alpha + beta);
	}

	double AccuracyFlex::GetBayesError(int n, int e, int depth) const {
		double rho0 = 2.5;
		double rho1 = 2.5;
		double alpha = 0.95;
		double beta = 0.5;

		double psplit = 0;//std::log(alpha * std::pow(1 + depth, -beta));
		double error = -psplit - logBeta(e + rho0, (n - e) + rho1) + logBeta(rho0, rho1);
		return error;

	}

	double AccuracyFlex::GetMLossError(int n, int e) const {
		if (e == 0 || e == n) return 0;
		double yhat = ((double)e) / ((double)n);
		double y = e > n / 2 ? 1 : 0;
		return n * (y / yhat + (1 - y) / (1 - yhat) - 1);
	}

	double AccuracyFlex::GetLLossError(int n, int e) const {
		if (e == 0 || e == n) return 0;
		double yhat = ((double)e) / ((double)n);
		double y = e > n / 2 ? 1 : 0;
		return n * (y / std::sqrt(1 - (1 - yhat) * (1 - yhat)) + (1 - y) / std::sqrt(1 - yhat * yhat) - 1);
	}

	Node<AccuracyFlex> AccuracyFlex::SolveLeafNode(const ADataView& data, const BranchContext& context) const {
		int opt_label = 0;
		int opt_count = data.NumInstancesForLabel(0);
		for (int k = 0; k < data.NumLabels(); k++) {
			if (data.NumInstancesForLabel(k) > opt_count) {
				opt_label = k;
				opt_count = data.NumInstancesForLabel(k);
			}
		}
		return Node<AccuracyFlex>(opt_label, GetLeafCosts(data, context, opt_label));
	}

	double AccuracyFlex::GetObjectiveError(int n, int e, int label, int depth) const {
		switch (objective_type) {
		case AccuracyObjective::misclassification_score:
			return GetMisclassificationScore(n, e, label);
		case AccuracyObjective::entropy:
			return GetEntropy(n, e);
		case AccuracyObjective::gini_index:
			return GetGiniScore(n, e);
		case AccuracyObjective::sqrt_gini:
			return GetSqrtGiniScore(n, e);
		case AccuracyObjective::minimum_error:
			return GetMinError(n, e);
		case AccuracyObjective::mdl_quinlan:
			return GetMDLQuinlan(n, e);
		case AccuracyObjective::mdl_mehta:
			return GetMDLMehta(n, e);
		case AccuracyObjective::pessimistic_binomial:
			return  e + GetPessimisticErrorIncrease(n, e);
		case AccuracyObjective::bayes:
			return GetBayesError(n, e, depth);
		case AccuracyObjective::mloss:
			return GetMLossError(n, e);
		case AccuracyObjective::lloss: 
			return GetLLossError(n, e);
		}
		return 0;
	}

	double AccuracyFlex::GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const {
		return GetObjectiveError(
			data.Size() + smoothing * 2, 
			GetError(data, label) + smoothing, 
			label, context.GetBranch().Depth());
	}

	int AccuracyFlex::GetTestLeafCosts(const ADataView& data, const BranchContext& context, int label) const {
		return GetError(data, label);
	}

	void AccuracyFlex::ComputeD2Costs(const int& d2costs, int count, int label, int depth, double& costs) const {
		if (d2costs > 0.5 * count) costs = INT32_MAX;
		else costs = GetObjectiveError(count + smoothing * 2, d2costs + smoothing, label, depth);
	}

	void GetIntLinSpace(int start, int end_exclusive, int count, std::vector<int>& nums) {
		nums.clear();
		if (count == 0) return;
		// Length is the maximum number of configs that can be selected
		int length = end_exclusive - start;
		count = std::min(count, length);
		double step = std::max(double(length) / (count + 1), 1.0);
		double end = end_exclusive - 1.0;
		for (int i = 1; i <= count; i++) {
			nums.push_back(std::max(start, int(std::ceil(end - (count - i) * step))));
		}
	}

	void GetLinSpace(double start, double end, int count, std::vector<double>& nums) {
		nums.clear();
		if (count == 0) return;
		// Length is the maximum number of configs that can be selected
		double length = end - start;
		double step = length / (count - 1);
		for (int i = 1; i <= count; i++) {
			nums.push_back(std::max(start, end - (count - i) * step));
		}
	}

	void GetLogSpace(double start, double end, int count, std::vector<double>& nums) {
		nums.clear();
		if (count == 0) return;
		// Length is the maximum number of configs that can be selected
		double log_start = std::log(start);
		double log_end   = std::log(end);
		std::vector<double> log_nums;
		GetLinSpace(log_start, log_end, count, log_nums);
		for (int i = 0; i < count; i++) {
			nums.push_back(std::exp(log_nums[i]));
		}
	}

	TuneRunConfiguration AccuracyFlex::GetTuneRunConfiguration(const ParameterHandler& default_config, const ADataView& data, int phase) {
		auto tune_method = GetAccuracyTuningMethod(default_config);
		const int num_configurations = int(default_config.GetIntegerParameter("num-hyper-params"));
		switch (tune_method) {
		case AccuracyTuningMethod::tree_size:
		{
			ParameterHandler params = default_config;
			params.SetFloatParameter("cost-complexity", 0.0);
			auto tune_config = OptimizationTask::GetTuneRunConfiguration(params, data, phase);
			if (tune_config.GetNumberOfConfigs() < num_configurations) return tune_config;
			std::vector<int> selected_configs;
			if (num_configurations > 1) {
				/*std::vector<int> selected_configs;
				GetIntLinSpace(0, tune_config.GetNumberOfConfigs(), num_configurations, selected_configs);
				*/
				std::vector<double> d_selected_configs;
				GetLogSpace(1.0, double(tune_config.GetNumberOfConfigs()), num_configurations, d_selected_configs);

				int prev = -1;
				for (double d_sc : d_selected_configs) {
					int sc = int(std::round(d_sc - 1.0));
					if (sc <= prev) sc = prev + 1;
					selected_configs.push_back(sc);
					prev = sc;
				}
			} else {
				selected_configs.push_back(tune_config.GetNumberOfConfigs() - 1);
			}
			TuneRunConfiguration config;
			for (int i: selected_configs) {
				config.AddConfiguration(tune_config.parameters[i], tune_config.descriptors[i]);
			}
			return config;
		}
		case AccuracyTuningMethod::depth:
		{
			TuneRunConfiguration config;
			int max_nodes = int(default_config.GetIntegerParameter("max-num-nodes"));
			int max_d = int(default_config.GetIntegerParameter("max-depth"));
			std::vector<int> selected_configs;
			GetIntLinSpace(0, max_d+1, num_configurations, selected_configs);

			for (int d: selected_configs) {
				int num_nodes = std::min(max_nodes, (1 << d) - 1);
				ParameterHandler params = default_config;
				params.SetIntegerParameter("max-depth", d);
				params.SetIntegerParameter("max-num-nodes", num_nodes);
				params.SetFloatParameter("cost-complexity", 0.0);
				config.AddConfiguration(params, "d=" + std::to_string(d) + ", n=" + std::to_string(num_nodes));
			}
			config.reset_solver = false;
			return config;
		}
		case AccuracyTuningMethod::cc:
		{
			TuneRunConfiguration config;
			double base_alpha = 1.0 / data.Size();
			std::vector<double> alphas;
			GetLogSpace(base_alpha, 0.1, num_configurations - 1, alphas);
			std::reverse(alphas.begin(), alphas.end());
			alphas.push_back(0.0);
			for (auto a : alphas) {
				ParameterHandler params = default_config;
				params.SetFloatParameter("cost-complexity", a);
				config.AddConfiguration(params, "a = " + std::to_string(a));
			}
			config.reset_solver = true;
			config.skip_when_max_tree = true;
			return config;
		}
		case AccuracyTuningMethod::wcc:
		{
			TuneRunConfiguration config;
			double base_alpha = 1.0 / data.Size();
			std::vector<double> alphas;
			GetLogSpace(base_alpha, 0.1, num_configurations - 1, alphas);
			std::reverse(alphas.begin(), alphas.end());
			alphas.push_back(0.0);
			for (auto a : alphas) {
				ParameterHandler params = default_config;
				params.SetFloatParameter("cost-complexity", a);
				config.AddConfiguration(params, "a = " + std::to_string(a));
			}
			config.reset_solver = true;
			config.skip_when_max_tree = true;
			return config;
		}
		case AccuracyTuningMethod::min_support:
		{
			TuneRunConfiguration config;
			double base_alpha = 1.0 / data.Size();
			std::vector<double> alphas;
			GetLogSpace(base_alpha, 0.2, num_configurations, alphas);
			std::vector<int> node_sizes;
			int prev = 0;
			for (double a : alphas) {
				int node_size = int(std::round(a * data.Size()));
				if (node_size <= prev) node_size = prev + 1;
				node_sizes.push_back(node_size);
				prev = node_size;
			}
			std::sort(node_sizes.begin(), node_sizes.end(), std::greater<int>());
			for (auto ns : node_sizes) {
				ParameterHandler params = default_config;
				params.SetIntegerParameter("min-leaf-node-size", ns);
				config.AddConfiguration(params, "ms = " + std::to_string(ns));
			}
			config.reset_solver = true;
			config.skip_when_max_tree = true;
			return config;
		}
		case AccuracyTuningMethod::smoothing:
		{
			TuneRunConfiguration config;

			double base_alpha = 1.0 / data.Size();
			std::vector<double> alphas;
			GetLogSpace(base_alpha, 0.05, num_configurations - 1, alphas);
			std::vector<int> smoothing_counts = { 0 };
			int prev = 0;
			for (double a : alphas) {
				int smoothing_count = int(std::round(a * data.Size()));
				if (smoothing_count <= prev) smoothing_count = prev + 1;
				smoothing_counts.push_back(smoothing_count);
				prev = smoothing_count;
			}
			std::sort(smoothing_counts.begin(), smoothing_counts.end(), std::greater<int>());
			for (auto sc : smoothing_counts) {
				ParameterHandler params = default_config;
				params.SetIntegerParameter("smoothing", sc);
				config.AddConfiguration(params, "sc = " + std::to_string(sc));
			}
			config.reset_solver = true;
			config.skip_when_max_tree = false;
			return config;
		}
		default:
			std::cout << "Provide a valid value for the tune-method." << std::endl;
			exit(1);
		}
	}

}