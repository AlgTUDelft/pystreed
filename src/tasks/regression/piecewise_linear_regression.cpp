#include "tasks/regression/piecewise_linear_regression.h"
#include "solver/similarity_lowerbound.h"
#include "solver/tree.h"

namespace STreeD {

	PieceWiseLinearRegExtraData PieceWiseLinearRegExtraData::ReadData(std::istringstream& iss, int num_extra_cols) {
		std::vector<double> continuous_features;
		for (int i = 0; i < num_extra_cols; i++) {
			double f;
			iss >> f;
			continuous_features.push_back(f);
		}
		return { continuous_features };
	}

	inline double GetCF(const AInstance* instance, int cf_ix) {
		return static_cast<const Instance<double, PieceWiseLinearRegExtraData>*>(instance)->GetExtraData().x[cf_ix];
	}

	void SetCF(AInstance* instance, int cf_ix, double val) {
		static_cast<Instance<double, PieceWiseLinearRegExtraData>*>(instance)->GetMutableExtraData().x[cf_ix] = val;
	}

	int GetNumCF(const ADataView& data) {
		return int(static_cast<const Instance<double, PieceWiseLinearRegExtraData>*>
			(data.GetInstancesForLabel(0)[0])->GetExtraData().x.size());
	}

	int GetNumCF(const AInstance* instance) {
		return int(static_cast<const Instance<double, PieceWiseLinearRegExtraData>*>
			(instance)->GetExtraData().x.size());
	}

	double GetStd(double v_sq_sum, double v_sum, size_t n) {
		return std::sqrt(v_sq_sum / n - v_sum * v_sum / (n * n));
	}

	const LinearModel PieceWiseLinearRegression::worst_label = LinearModel();

	bool LinearModel::operator==(const LinearModel& other) const {
		if (std::abs(b0 - other.b0) > 1e-6) return false;
		if (b.size() != other.b.size()) return false;
		for (int j = 0; j < b.size(); j++) {
			if (std::abs(b[j] - other.b[j]) > 1e-6) return false;
		}
		return true;
	}

	double LinearModel::Predict(const AInstance* instance) const {
		double y = b0;
		for (int j = 0; j < b.size(); j++) {
			y += b[j] * GetCF(instance, j);
		}
		return y;
	}

	std::string LinearModel::ToString() const {
		std::stringstream ss;
		bool empty = true;
		if (std::abs(b0) > 1e-6) {
			empty = false;
			ss << b0;
		}
		for (int i = 0; i < b.size(); i++) {
			if (std::abs(b[i]) > 1e-6) {
				if (!empty && b[i] > 0) ss << " + ";
				else if(b[i] < 0) ss << " - ";
				ss << std::abs(b[i]) << "x" << (i + 1);
				empty = false;
			}
		}
		if (empty) return "0";
		return ss.str();
	}

	void PieceWiseLinearRegression::PreprocessData(AData& data, bool train) { }

	void PieceWiseLinearRegression::PreprocessTrainData(ADataView& train_data) {
	
		std::vector<const AInstance*>& instances = train_data.GetMutableInstancesForLabel(0);
		size_t n = instances.size();
		int n_cf = GetNumCF(instances[0]);

		double y = 0;
		double yy = 0;
		for (auto i : instances) {
			double label = GetInstanceLabel<double>(i);
			y += label;
			yy += label * label;
		}

		y_mu = 0;//y / n;
		y_std = 1;//GetStd(yy, y, n);

		////////////////////////////////////////////////
		// Normalize continuous features ///////////////
		////////////////////////////////////////////////
		std::vector<double> cf_y(n_cf, 0.0), cf_ysq(n_cf, 0.0);

		for (auto i : instances) {
			for (int j = 0; j < n_cf; j++) {
				double cf = GetCF(i, j);
				cf_y[j] += cf;
				cf_ysq[j] += cf * cf;
			}
		}
		g_cf_mu.resize(n_cf);
		g_cf_std.resize(n_cf);
		for (int j = 0; j < n_cf; j++) {
			g_cf_mu[j] = cf_y[j] / n; // The mean for this continuous feature
			g_cf_std[j] = GetStd(cf_ysq[j], cf_y[j], n); // The standard deviation for this feature
		}


		for (int ix = 0; ix < instances.size(); ix++) {
			auto& i = instances[ix];
			double label = GetInstanceLabel<double>(i);
			auto copy_instance = new Instance<double, PieceWiseLinearRegExtraData>(*static_cast<const Instance<double, PieceWiseLinearRegExtraData>*>(i));
			auto reg_i = static_cast<Instance<double, PieceWiseLinearRegExtraData>*>(copy_instance);
			SetInstanceLabel<double>(reg_i, (label - y_mu) / y_std);
			for (int j = 0; j < n_cf; j++) {
				if (std::abs(g_cf_std[j]) < 1e-6) continue;
				SetCF(reg_i, j, (GetCF(i, j) - g_cf_mu[j]) / g_cf_std[j]);
			}
			instances[ix] = reg_i;
			data.AddInstance(reg_i);
		}	
	}

	void PieceWiseLinearRegression::PostProcessTree(std::shared_ptr<Tree<PieceWiseLinearRegression>> tree) {
		if (tree->IsLabelNode()) {
			// reverse normalize the tree
			auto& model = tree->label;
			int CF = int(g_cf_mu.size());
			std::vector<double> b(CF);
			for (int j = 0; j < CF; j++) {
				b[j] = model.b[j] / g_cf_std[j];
			}
			double b0 = model.b0;
			for (int j = 0; j < CF; j++) {
				b0 -= g_cf_mu[j] * b[j];
			}
			for (int j = 0; j < CF; j++) {
				b[j] *= y_std;
			}
			b0 = b0 * y_std + y_mu;
			model.b0 = b0;
			model.b = b;
		} else {
			PostProcessTree(tree->left_child);
			PostProcessTree(tree->right_child);
		}
	}

	void PieceWiseLinearRegression::InformTrainData(const ADataView& train_data, const DataSummary& train_summary) {
		OptimizationTask::InformTrainData(train_data, train_summary);
		const std::vector<const AInstance*>& instances = train_data.GetInstancesForLabel(0);
		const int N = int(instances.size());
		const int CF = GetNumCF(train_data);

		max_lb = std::abs(GetInstanceLabel<double>(instances[0]));
		
		double ys = 0;
		double yys = 0;
		double ys_scaled = 0;
		double yys_scaled = 0;

		for (auto i : instances) {
			auto instance = static_cast<const Instance<double, PieceWiseLinearRegExtraData>*>(i);
			double label = instance->GetLabel();
			if (std::abs(label) > max_lb) max_lb = std::abs(label);
			double weight = i->GetWeight();
			ys_scaled += weight * label;
			yys_scaled += weight * label * label;
			
			double org_label = (label + y_mu) * y_std;
			ys += weight * org_label;
			yys += weight * org_label * org_label;
		}

		// Scale cost complexity so it has approximately the same meaning for different datasets
		cost_complexity_norm_coef = yys - (ys * ys / N);
		double scaled_cost_complexity_norm_coef = yys_scaled - (ys_scaled * ys_scaled / N);
		branching_cost = cost_complexity_parameter * scaled_cost_complexity_norm_coef;
		runtime_assert(branching_cost >= 0);
		
		max_lb += std::sqrt(scaled_cost_complexity_norm_coef);
		

		// Initialize the vectors for GLMNet
		X = std::vector<std::vector<double>>(CF, std::vector<double>(N, 0.0));
		x1x2 = std::vector<std::vector<double>>(CF, std::vector<double>(CF, 0.0));
		//y.resize(N);
		cf_y.resize(CF);
		cf_ysq.resize(CF);
		cf_mu.resize(CF);
		cf_std.resize(CF);
		b.resize(CF);
		x.resize(CF);
		xy.resize(CF);
		//xx.resize(CF);
		redundant_feature.resize(CF);
		is_non_zero.resize(CF);

	}

	void PieceWiseLinearRegression::InformTestData(const ADataView& test_data, const DataSummary& test_summary) {
		OptimizationTask::InformTestData(test_data, test_summary);
		// Compute test total variance
		double test_ys = 0;
		double test_yys = 0;
		for (auto i : test_data.GetInstancesForLabel(0)) {
			auto instance = static_cast<const Instance<double, PieceWiseLinearRegExtraData>*>(i);
			double label = instance->GetLabel();
			double ysq = label * label;
			double weight = i->GetWeight();
			test_ys += weight * label;
			test_yys += weight * ysq;
		}
		test_total_variance = test_yys - (test_ys * test_ys / test_data.Size());
	}

	inline double GetCovariate(const std::vector<std::vector<double>>& X, std::vector<std::vector<double>>& x1x2, const size_t N, const int j, const int k) {
		auto& _x1x2 = x1x2[j][k];
		if (std::abs(_x1x2) < 1e-4) {
			auto& xj = X[j];
			auto& xk = X[k];
			for (size_t i = 0; i < N; i++) {
				_x1x2 += xj[i] * xk[i];
			}
			x1x2[k][j] = _x1x2;
		}
		return _x1x2;
	}

	std::pair<LinearModel,  double> PieceWiseLinearRegression::SolveGLMNet(const ADataView& data) const {
		//Initialize variables
		const int CF = GetNumCF(data); // Number of continuous features
		const size_t N = data.Size();  // Number of data instances
		double prev_sse = DBL_MAX;	   // Previous SSE
		double b0 = 0;					// Intercept
		// Initialize b coefficients vector with 0's
		std::fill(b.begin(), b.end(), 0.0);
		auto& instances = data.GetInstancesForLabel(0);
		// Covariances
		for (int j = 0; j < CF; j++) {
			std::fill(x1x2[j].begin(), x1x2[j].end(), 0.0);
		}
		// list of features with non-zero values in b
		std::vector<int> non_zeros;
		non_zeros.reserve(CF);
		std::fill(is_non_zero.begin(), is_non_zero.end(), 0);
		// Keep track of features that are redundant (all same value)
		std::fill(redundant_feature.begin(), redundant_feature.end(), 0);

		// Compute normalization for Y
		double _y_sum = 0;
		double _ysq_sum = 0;
		std::fill(cf_y.begin(), cf_y.end(), 0.0);
		std::fill(cf_ysq.begin(), cf_ysq.end(), 0.0);
		double label;
		for (auto& inst : instances) {
			label = GetInstanceLabel<double>(inst);
			_y_sum += label;
			_ysq_sum += label * label;
		}
		double y_mu = _y_sum / N;
		double y_std = 1.0;
		
		// Compute normalization for X
		double f = 0;
		double prev_f;
		for (int j = 0; j < CF; j++) {
			auto& _cf_y = cf_y[j];
			auto& _cf_ysq = cf_ysq[j];
			auto& _X = X[j];
			prev_f = GetCF(instances[0], j);
			double delta_sum = 0;
			int i = 0;
			for (auto& inst : instances) {
				f = GetCF(inst, j);
				_cf_y += f;
				_cf_ysq += f * f;
				delta_sum += std::abs(f - prev_f);
				_X[i++] = f;
				prev_f = f;
			}
			// Compute mu for X to center the X features
			cf_mu[j] = cf_y[j] / N;
			cf_std[j] = 1.0;
			// Mark as redundant if the difference between feature values is close to zero.
			redundant_feature[j] = std::abs(delta_sum) < 1e-4 ? 1 : 0; 
		}
			
		// Center all X values by their mu
		for (int j = 0; j < CF; j++) {
			if (redundant_feature[j]) {
				std::fill(X[j].begin(), X[j].end(), 0.0);
				continue;
			}
			double mu = cf_mu[j];
			auto& _X = X[j];
			for (auto& _x: _X) {
				_x = (_x - mu);
			}
		}

		// Precompute sums for every j
		std::fill(xy.begin(), xy.end(), 0.0);
		std::fill(x.begin(), x.end(), 0.0);
		//std::fill(xx.begin(), xx.end(), 0.0);
		double ysq_sum = 0;
		double y_sum = 0;
		for (int j = 0; j < CF; j++) {
			if (redundant_feature[j]) continue;
			auto& _X = X[j];
			auto& _xy = xy[j];
			auto& _x = x[j];
			auto& _xx = x1x2[j][j];
			for (size_t i = 0; i < N; i++) {
				label = GetInstanceLabel<double>(instances[i]) - y_mu;
				ysq_sum += label * label;
				y_sum += label;
				f = _X[i];
				_xy += f * label;
				_x += f;
				_xx += f * f;
			}
		}

		double lambda_max = 0;
		for (int j = 0; j < CF; j++) {
			lambda_max = std::max(lambda_max, std::abs(xy[j]));
		}
		double lambda = N * lasso_penalty;
		double gamma = N * ridge_penalty;

		if (lambda >= lambda_max) {
			return std::make_pair(LinearModel(), DBL_MAX);
		}
		
		b0 = 0; // Intercept should be zero after normalization

		int iteration = 0;
		bool significant_change = true;

		double old_bj, bj, zj;
		int j;
		while (significant_change && iteration++ < 10000) {
			significant_change = false;
			for (j = 0; j < CF; j++) {
				if (redundant_feature[j]) continue;
				old_bj = b[j];
				bj = xy[j];
				zj = x1x2[j][j];
				for (int &k : non_zeros) {
					if (k == j) continue;
					bj -= b[k] * GetCovariate(X, x1x2, N, j, k);
				}
				//bj += b[j] * zj;
				
				if (std::abs(bj) < lambda) continue;
				if (bj < 0) {
					b[j] = (bj + lambda) / (zj + gamma);
				} else {
					b[j] = (bj - lambda) / (zj + gamma);
				}
				//b[j] = (bj < 0 ? -1 : 1) * std::max(std::abs(bj) - lambda, 0.0) / (zj + gamma);
				
				if (std::abs((b[j] - old_bj) / old_bj) > 1e-3) // If more than 0.1% change
					significant_change = true;

				if (!is_non_zero[j]) {
					is_non_zero[j] = 1;
					non_zeros.push_back(j);
				}
			}

			// Compute error SSE = Y^T x Y - 2 B^T x X^T x Y + B^T x X^T x X x B
			double sse = ysq_sum - 2 * b0 * y_sum + b0 * b0 * N;
			double b_l1_norm = 0;
			double b_l2_norm_sq = 0;
			for (int& k : non_zeros) {
				sse -= 2 * b[k] * xy[k];
				b_l1_norm += std::abs(b[k]);
				b_l2_norm_sq += b[k] * b[k];
			}
			for (int j = 0; j < CF; j++) {
				if (!is_non_zero[j]) continue;
				sse += 2 * b0 * b[j] * x[j];
				sse += b[j] * b[j] * x1x2[j][j];
				for (int k = j + 1; k < CF; k++) {
					if (!is_non_zero[k]) continue;
					sse += 2 * b[j] * b[k] * GetCovariate(X, x1x2, N, j, k);
				}
			}
			// Add Lasso penalty. Lasso normally optimizes 1/2 MSE + lambda * ||b||
			// Since we compute SSE instead of 1/2 MSE, the penalty becomes 2 * N * lambda * ||b||
			// N was already included by not dividing lambda by N above
			sse += 2 * lambda * b_l1_norm;
			// Add Ridge penalty
			sse += gamma * b_l2_norm_sq;
			
			if (!(prev_sse > 0.9 * DBL_MAX) && (prev_sse - sse) / sse < 1e-3) break;
			prev_sse = sse;

		}
		
		// Reverse Normalize
		std::vector<double> _b(CF);
		for (int j = 0; j < CF; j++) {
			_b[j] = b[j] / cf_std[j];
		}
		double _b0 = y_mu;
		for (int j = 0; j < CF; j++) {
			_b0 -= cf_mu[j] * _b[j];
		}

		// Compute normalized SSE = Y^T x Y - 2 B^T x X^T x Y + B^T x X^T x X x B
		// Here only return the SSE, do not return the Lasso penalty
		double sse = _ysq_sum - 2 * _b0 * _y_sum + _b0 * _b0 * N;
		double b_l1_norm = 0;
		double b_l2_norm_sq = 0;
		for (int k : non_zeros) {
			double _xy = xy[k] * y_std * cf_std[k] + x[k] * cf_std[k] * y_mu + cf_mu[k] * y_sum * y_std + N * cf_mu[k] * y_mu;
			sse -= 2 * _b[k] * _xy;
			b_l1_norm += std::abs(_b[k]);
			b_l2_norm_sq += _b[k] * _b[k];
		}
		for (int j = 0; j < CF; j++) {
			if (!is_non_zero[j]) continue;
			sse += 2 * _b0 * _b[j] * cf_y[j];
			sse += _b[j] * _b[j] * cf_ysq[j];
			for (int k = j + 1; k < CF; k++) {
				if (std::abs(_b[k]) < 1e-4) continue;
				double _x1x2 = GetCovariate(X, x1x2, N, j, k) * cf_std[j] * cf_std[k]
					+ x[j] * cf_std[j] * cf_mu[k] + x[k] * cf_std[k] * cf_mu[j] + N * cf_mu[j] * cf_mu[k];
				sse += 2 * _b[j] * _b[k] * _x1x2;
			}
		}
		sse += 2 * b_l1_norm * lambda;
		sse += b_l2_norm_sq * gamma;
		
		return std::make_pair(LinearModel(_b, _b0), sse);
	}

	Node<PieceWiseLinearRegression> PieceWiseLinearRegression::SolveLeafNode(const ADataView& data, const BranchContext& context) const {
		// Reuse previous computation if the same
		if (context.GetBranch() == last_branch && last_branch.Depth() > 0 && last_solution.label != worst_label) {
			return last_solution;
		}
		
		auto result = SolveGLMNet(data);
		//auto result = SolvePiecewiseLinearRegressionWithDeterminant(data);		
		// Store computation for reuse
		last_branch = context.GetBranch();
		last_solution = Node<PieceWiseLinearRegression>(result.first, result.second);
		return last_solution;

	}

	double PieceWiseLinearRegression::GetLeafCosts(const ADataView& data, const BranchContext& context, const LinearModel& model) const {
		double error = 0;
		for (auto& i : data.GetInstancesForLabel(0)) {
			double ytrue = GetInstanceLabel<double>(i);
			double ypred = model.Predict(i);
			error += std::pow(ytrue - ypred, 2);
		}
		return error;
	}

	double PieceWiseLinearRegression::GetTestLeafCosts(const ADataView& data, const BranchContext& context, const LinearModel& model) const {
		double error = 0;
		for (auto& i : data.GetInstancesForLabel(0)) {
			double ytrue = (GetInstanceLabel<double>(i) + y_mu) * y_std;
			double ypred = (model.Predict(i) + y_mu) * y_std;
			error += std::pow(ytrue - ypred, 2);
		}
		return error;
	}

	PairWorstCount<PieceWiseLinearRegression> PieceWiseLinearRegression::ComputeSimilarityLowerBound(const ADataView& data_old, const ADataView& data_new) const {
		int total_diff = 0;
		double worst_diff = 0;
		for (int label = 0; label < data_new.NumLabels(); label++) {
			auto& new_instances = data_new.GetInstancesForLabel(label);
			auto& old_instances = data_old.GetInstancesForLabel(label);
			int size_new = int(new_instances.size());
			int size_old = int(old_instances.size());
			int index_new = 0, index_old = 0;
			while (index_new < size_new && index_old < size_old) {
				int id_new = new_instances[index_new]->GetID();
				int id_old = old_instances[index_old]->GetID();
				int weight_new = int(new_instances[index_new]->GetWeight());
				int weight_old = int(old_instances[index_old]->GetWeight());
				//the new data has something the old one does not
				if (id_new < id_old) {
					total_diff += weight_new;
					index_new++;
				}
				//the old data has something the new one does not
				else if (id_new > id_old) {
					total_diff += weight_old;
					worst_diff += weight_old * GetWorstPerLabel(GetInstanceLabel<double>(old_instances[index_old]));
					index_old++;
				} else {//no difference
					index_new++;
					index_old++;
				}
			}


			for (; index_new < size_new; index_new++) {
				int weight_new = int(new_instances[index_new]->GetWeight());
				total_diff += weight_new;
			}
			for (; index_old < size_old; index_old++) {
				int weight_old = int(old_instances[index_old]->GetWeight());
				total_diff += weight_old;
				worst_diff += weight_old * GetWorstPerLabel(GetInstanceLabel<double>(old_instances[index_old]));
			}
		}
		PairWorstCount<PieceWiseLinearRegression> result(worst_diff, total_diff);
		return result;
	}

	/*
	* Hyper tuning
	*/

	double ComputeMaxLambda(const ADataView& data) {
		const std::vector<const AInstance*>& instances = data.GetInstancesForLabel(0);
		const int N = int(instances.size());
		const int CF = GetNumCF(data);
		
		// Compute XY sum
		std::vector<double> xy(CF, 0.0);

		for (auto i : instances) {
			auto instance = static_cast<const Instance<double, PieceWiseLinearRegExtraData>*>(i);
			double label = instance->GetLabel();
			for (int j = 0; j < CF; j++) {
				double f = GetCF(i, j);
				xy[j] += label * f;
			}
		}
		double lambda_max = 0.0;
		for (int j = 0; j < CF; j++) {
			lambda_max = std::max(lambda_max, std::abs(xy[j] / N));
		}
		return lambda_max;
	}

	TuneRunConfiguration PieceWiseLinearRegression::GetTuneRunConfiguration(const ParameterHandler& default_config, const ADataView& data, int tune_phase) {
		TuneRunConfiguration config;
		
		std::vector<double> alphas = { 0.1, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.0005, 0.0001 };
		std::vector<double> lasso_penalizations = { 0.5, 0.05, 0.005, 0 };
		std::vector<double> l1_ratios = { 0, 0.25, 0.5, 0.75, 1.0 };

		double max_lambda = ComputeMaxLambda(data);
		if (tune_phase == 0) {
			ParameterHandler d0_config = default_config;
			d0_config.SetFloatParameter("cost-complexity", .99);
			for (auto lp : lasso_penalizations) {
				for (auto l1_ratio : l1_ratios) {
					double a = l1_ratio * lp * max_lambda;
					double b = (1.0 - l1_ratio) * lp * max_lambda;
					ParameterHandler params = d0_config;
					params.SetFloatParameter("lasso-penalty", a);
					params.SetFloatParameter("ridge-penalty", b);
					config.AddConfiguration(params, "d0, lp = " + std::to_string(a) + ", rp = " + std::to_string(b));
					if (lp == 0.0) break;
				}
				
			}
			config.skip_when_max_tree = false;
		} else if (tune_phase == 1) {
			for (auto a : alphas) {
				ParameterHandler params = default_config;
				params.SetFloatParameter("cost-complexity", a);
				config.AddConfiguration(params, "a = " + std::to_string(a));
			}
			config.skip_when_max_tree = true;
		} else {
			for (auto lp : lasso_penalizations) {
				for (auto l1_ratio : l1_ratios) {
					double a = l1_ratio * lp * max_lambda;
					double b = (1.0 - l1_ratio) * lp * max_lambda;
					ParameterHandler params = default_config;
					params.SetFloatParameter("lasso-penalty", a);
					params.SetFloatParameter("ridge-penalty", b);
					config.AddConfiguration(params, "d0, lp = " + std::to_string(a) + ", rp = " + std::to_string(b));
					if (lp == 0.0) break;
				}
			}
			config.skip_when_max_tree = false;
		}
		config.reset_solver = true;
		config.runs = 5;
		return config;
	}

}