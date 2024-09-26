#include "tasks/regression/piecewise_simple_linear_regression.h"
#include "solver/similarity_lowerbound.h"

namespace STreeD {

	const LinearModel SimpleLinearRegression::worst_label = LinearModel();

	SimpleLinRegExtraData::SimpleLinRegExtraData(const std::vector<double>& x) : PieceWiseLinearRegExtraData(x) {
		int num_features = int(x.size());
		xsq.resize(num_features, 0.0);
		yx.resize(num_features, 0.0);
	}

	SimpleLinRegExtraData SimpleLinRegExtraData::ReadData(std::istringstream& iss, int num_extra_cols) {
		SimpleLinRegExtraData ed;
		std::vector<double>& continuous_features = ed.x;
		for (int i = 0; i < num_extra_cols; i++) {
			double f;
			iss >> f;
			continuous_features.push_back(f);
		}
		ed.xsq.resize(ed.NumContFeatures(), 0.0);
		ed.yx.resize(ed.NumContFeatures(), 0.0);
		return ed;
	}

	int GetNumContFeatures(const ADataView& data) {
		return int(static_cast<const Instance<double, SimpleLinRegExtraData>*>
			(data.GetInstancesForLabel(0)[0])->GetExtraData().NumContFeatures());
	}

	int GetNumContFeatures(const AInstance* instance) {
		return int(static_cast<const Instance<double, SimpleLinRegExtraData>*>
			(instance)->GetExtraData().NumContFeatures());
	}
	
	
	const D2SimpleLinRegSol& D2SimpleLinRegSol::operator+=(const D2SimpleLinRegSol& v2) {
		ys += v2.ys;
		yys += v2.yys;
		weight += v2.weight;
		if (v2.xs.size() == 0) return *this;
		if (xs.size() < v2.xs.size()) {
			xs.resize(v2.xs.size(), 0.0);
			xsq.resize(v2.xsq.size(), 0.0);
			yx.resize(v2.yx.size(), 0.0);
		}
		int size = int(v2.xs.size());
		// fast vector addition
		double* __restrict _xs = &xs[0];
		double* __restrict _xsq = &xsq[0];
		double* __restrict _yx = &yx[0];
		const double* __restrict _v2_xs = &(v2.xs[0]);
		const double* __restrict _v2_xsq = &(v2.xsq[0]);
		const double* __restrict _v2_yx = &(v2.yx[0]);
		while (size--) {
			(*_xs++) += (*_v2_xs++);
			(*_xsq++) += (*_v2_xsq++);
			(*_yx++) += (*_v2_yx++);
		}
		return *this;
	}

	const D2SimpleLinRegSol& D2SimpleLinRegSol::operator-=(const D2SimpleLinRegSol& v2) {
		ys -= v2.ys;
		yys -= v2.yys;
		weight -= v2.weight;
		if (v2.xs.size() == 0) return *this;
		if (xs.size() < v2.xs.size()) {
			xs.resize(v2.xs.size(), 0.0);
			xsq.resize(v2.xsq.size(), 0.0);
			yx.resize(v2.yx.size(), 0.0);
		}
		int size = int(v2.xs.size());
		// fast vector subtraction
		double* __restrict _xs = &xs[0];
		double* __restrict _xsq = &xsq[0];
		double* __restrict _yx = &yx[0];
		const double* __restrict _v2_xs = &(v2.xs[0]);
		const double* __restrict _v2_xsq = &(v2.xsq[0]);
		const double* __restrict _v2_yx = &(v2.yx[0]);
		while (size--) {
			(*_xs++) -= (*_v2_xs++);
			(*_xsq++) -= (*_v2_xsq++);
			(*_yx++) -= (*_v2_yx++);
		}
		return *this;
	}

	bool D2SimpleLinRegSol::operator==(const D2SimpleLinRegSol& v2) const {
		if (weight != v2.weight) return false;
		if (xs.size() != v2.xs.size()) return false;
		if (std::abs(ys - v2.ys) >= 1e-6) return false;
		if (std::abs(yys - v2.yys) >= 1e-6) return false;
		int num_f = int(xs.size());
		for (int f = 0; f < num_f; f++) {
			if (std::abs(xs[f] - v2.xs[f]) >= 1e-6) return false;
			if (std::abs(xsq[f] - v2.xsq[f]) >= 1e-6) return false;
			if (std::abs(yx[f] - v2.yx[f]) >= 1e-6) return false;
		}
		return true;
	}

	void SimpleLinearRegression::PreprocessData(AData& data, bool train) {
		std::vector<AInstance*>& instances = data.GetInstances();
		int n = int(instances.size());
		int num_cont_features = GetNumContFeatures(data.GetInstance(0));

		if (train) {
			// Arbitrary ordering by features, so that instances with equivalent features are next to each other
			// This sort only has to happen once. After a split there might be newly equivalent points, but since
			// the feature that is split on will be removed from consideration, and positive and negative instances
			// are mutually exclusive in the new DataViews, newly equivalent points will still be consecutive.
			std::sort(instances.begin(), instances.end(), [](const AInstance* a, const AInstance* b) {
				return a->GetFeatures() > b->GetFeatures();
			});

			int id = 0;
			double y = 0;
			double yy = 0;
			for (auto i : instances) {
				double label = GetInstanceLabel<double>(i);
				y += label;
				yy += label * label;
				i->SetID(id++);
			}
		}

		for (auto i : instances) {
			LInstance<double>* li = static_cast<LInstance<double>*>(i);
			double label = li->GetLabel();
			auto reg_i = static_cast<Instance<double, SimpleLinRegExtraData>*>(i);
			auto& ed = reg_i->GetMutableExtraData();
			ed.ysq = label * label;
			for (int f = 0; f < num_cont_features; f++) {
				ed.xsq[f] = ed.x[f] * ed.x[f];
				ed.yx[f] = label * ed.x[f];
			}
		}
	}

	void SimpleLinearRegression::PreprocessTrainData(ADataView& train_data) {
		auto& instances = train_data.GetMutableInstancesForLabel(0);
		int num_cont_features = GetNumContFeatures(train_data);

		std::sort(instances.begin(), instances.end(), [](const AInstance* a, const AInstance* b) {
			return a->GetID() < b->GetID();
		});


		// Get worst label distance for similarity_lower bound
		{
			min = GetInstanceLabel<double>(instances[0]) / instances[0]->GetWeight();
			max = min;
			double ys = 0;
			double yys = 0;
			total_training_weight = 0;
			for (auto i : instances) {
				auto instance = static_cast<const Instance<double, SimpleLinRegExtraData>*>(i);
				int weight = int(i->GetWeight());
				double label = instance->GetLabel();
				double ysq = instance->GetExtraData().ysq;
				if (label / weight < min) min = label / weight;
				if (label / weight > max) max = label / weight;
				ys += label;
				yys += ysq;
				total_training_weight += weight;
			}

			// Scale cost complexity so it has approximately the same meaning for different datasets
			double cost_complexity_norm_coef = yys - (ys * ys / total_training_weight);
			branching_cost = cost_complexity_parameter * cost_complexity_norm_coef;
			runtime_assert(branching_cost >= 0);

			double worst = max - min;
			worst_distance_squared = worst * worst;
		}
		
		auto prev = instances[0];
		int ix_first = 0;
		double y = GetInstanceLabel<double>(prev);
		auto& ed = GetInstanceExtraData<double, SimpleLinRegExtraData>(prev);
		double ysq = ed.ysq;
		std::vector<double> xs = ed.x;
		std::vector<double> xsq = ed.xsq;
		std::vector<double> yx = ed.yx;
		int w = int(prev->GetWeight());
		int n = w;

		//int total_weight = n;

		std::vector<double> delta_xs, delta_xsq, delta_yx;
		for (int i = 1; i < instances.size(); i++) {
			auto current = instances[i];
			
			double label = GetInstanceLabel<double>(current);
			auto& ed = GetInstanceExtraData<double, SimpleLinRegExtraData>(current);
			double delta_ysq = ed.ysq;
			delta_xs = ed.x;
			delta_xsq = ed.xsq;
			delta_yx = ed.yx;

			if (!prev->GetFeatures().HasEqualFeatures(current->GetFeatures())) {
				// If different
				if (n > w) {
					auto copy_instance = new Instance<double, SimpleLinRegExtraData>(*static_cast<const Instance<double, SimpleLinRegExtraData>*>(instances[ix_first]));
					SetInstanceLabel<double>(copy_instance, y); 
					auto reg_i = static_cast<Instance<double, SimpleLinRegExtraData>*>(copy_instance);
					auto& ed = reg_i->GetMutableExtraData();
					ed.ysq = ysq;
					for (int f = 0; f < num_cont_features; f++) {
						ed.x[f] = xs[f];
						ed.xsq[f] = xsq[f];
						ed.yx[f] = yx[f];
					}
					copy_instance->SetWeight(n);
					instances[ix_first] = copy_instance;
					data.AddInstance(copy_instance);
				} 
				instances[++ix_first] = current; // Swap instances, (to prevent having to do an expensive erase operation)
				y = 0;
				ysq = 0;
				n = 0;
				std::fill(xs.begin(), xs.end(), 0.0);
				std::fill(xsq.begin(), xsq.end(), 0.0);
				std::fill(yx.begin(), yx.end(), 0.0);
			} 
			w = int(current->GetWeight());
			n += w;
			y += label;
			ysq += delta_ysq;
			for (int f = 0; f < num_cont_features; f++) {
				xs[f] += delta_xs[f];
				xsq[f] += delta_xsq[f];
				yx[f] += delta_yx[f];
			}
			prev = current;
		}
		{
			auto mutable_instance = train_data.GetMutableInstance(0, ix_first);
			SetInstanceLabel<double>(mutable_instance, y);
			auto reg_i = static_cast<Instance<double, SimpleLinRegExtraData>*>(mutable_instance);
			auto& ed = reg_i->GetMutableExtraData();
			ed.ysq = ysq;
			for (int f = 0; f < num_cont_features; f++) {
				ed.x[f] = xs[f];
				ed.xsq[f] = xsq[f];
				ed.yx[f] = yx[f];
			}
			mutable_instance->SetWeight(n);
		}
		instances.resize(ix_first+1);
		train_data.ComputeSize();
	}

	void SimpleLinearRegression::InformTrainData(const ADataView& train_data, const DataSummary& train_summary) {
		OptimizationTask::InformTrainData(train_data,  train_summary);

		// Compute feature variances
		const std::vector<const AInstance*>& instances = train_data.GetInstancesForLabel(0);
		num_cont_features = GetNumContFeatures(instances[0]);
		feature_variance = std::vector<double>(num_cont_features);
		std::vector<double> xs(num_cont_features, 0.0);
		std::vector<double> xsq(num_cont_features, 0.0);

		for (auto i : instances) {
			auto instance = static_cast<const Instance<double, SimpleLinRegExtraData>*>(i);
			for (int j = 0; j < num_cont_features; j++) {
				xs[j] += instance->GetExtraData().x[j];
				xsq[j] += instance->GetExtraData().xsq[j];
			}
		}
		for (int j = 0; j < num_cont_features; j++) {
			feature_variance[j] = (xsq[j] - (xs[j] * xs[j] / total_training_weight)) / total_training_weight;
		}

	}

	void SimpleLinearRegression::InformTestData(const ADataView& test_data, const DataSummary& test_summary) {
		OptimizationTask::InformTestData(test_data, test_summary);

		// Compute test total variance
		double test_ys = 0;
		double test_yys = 0;
		for (auto i : test_data.GetInstancesForLabel(0)) {
			auto instance = static_cast<const Instance<double, SimpleLinRegExtraData>*>(i);
			double label = instance->GetLabel();
			double ysq = label * label;
			double weight = i->GetWeight();
			test_ys += label;
			test_yys += ysq;
		}
		test_total_variance = test_yys - (test_ys * test_ys / test_data.Size());
	}

	Node<SimpleLinearRegression> SimpleLinearRegression::SolveLeafNode(const ADataView& data, const BranchContext& context) const {
		// Reuse previous computation if the same
		if (context.GetBranch() == last_branch && last_branch.Depth() > 0 && last_solution.label != worst_label) {
			return last_solution;
		}

		double ysq = 0, y = 0, weight = 0;
		Node<SimpleLinearRegression> best_solution;
		if (data.Size() < 1) return best_solution;
		std::vector<double> xy(num_cont_features, 0.0), xsq(num_cont_features, 0.0), xs(num_cont_features, 0.0);
		for (const auto i : data.GetInstancesForLabel(0)) {
			auto instance = static_cast<const Instance<double, SimpleLinRegExtraData>*>(i);
			double w = i->GetWeight();
			double label = instance->GetLabel();
			auto& ed = instance->GetExtraData();
			ysq += ed.ysq;
			y += label;
			weight += w;

			// Fast vector addition
			double* __restrict _xy = &xy[0];
			double* __restrict _xsq = &xsq[0];
			double* __restrict _xs = &xs[0];
			const double* __restrict _ed_xy = &(ed.yx[0]);
			const double* __restrict _ed_xsq = &(ed.xsq[0]);
			const double* __restrict _ed_xs = &(ed.x[0]);
			int f = num_cont_features;
			while (f--) {
				(*_xy++) += (*_ed_xy++);
				(*_xsq++) += (*_ed_xsq++);
				(*_xs++) += (*_ed_xs++);
			}
			//for (int f = 0; f < num_cont_features; f++) {
			//	xy[f] += ed.yx[f];
			//	xsq[f] += ed.xsq[f];
			//	xs[f] += ed.x[f];
			//}
		}
		if (weight < minimum_leaf_node_size) return best_solution;
		double divisor, a, b, error{ 0 };
		for (int f = 0; f < num_cont_features; f++) {
			// y = a x + b
			divisor = (weight * xsq[f] - xs[f] * xs[f] + weight * feature_variance[f] * ridge_penalty);
			if (std::abs(divisor) < 1e-3) {
				a = 0;
				b = y / weight;
				error = ysq - (y * y / weight);
			} else {
				a = (weight * xy[f] - xs[f] * y) / divisor;
				b = (y - a * xs[f]) / weight;
				error = ysq - 2 * a * xy[f] - 2 * b * y + a * a * xsq[f] + 2 * a * b * xs[f] + weight * b * b
					+ ridge_penalty * a * a * feature_variance[f];
			}
			if (error < best_solution.solution) {
				best_solution.label = LinearModel(num_cont_features, f, a, b);
				best_solution.solution = std::max(0.0, error);
			}
		}
		return best_solution;
	}

	double SimpleLinearRegression::GetLeafCosts(const ADataView& data, const BranchContext& context, const LinearModel& model) const {
		double error = 0;
		int cf = 0;
		double max_coef = 0;
		for (int f = 1; f < num_cont_features; f++) {
			if (std::abs(model.b[f]) > max_coef) {
				cf = f;
				max_coef = std::abs(model.b[f]);
			}
		}
		double a = model.b[cf];
		double b = model.b0;
		for (auto& i : data.GetInstancesForLabel(0)) {
			auto instance = static_cast<const Instance<double, SimpleLinRegExtraData>*>(i);
			double w = i->GetWeight();
			double label = instance->GetLabel();
			auto& ed = instance->GetExtraData();
			error += ed.ysq - 2 * a * ed.yx[cf] - 2 * b * label
				+ a * a * ed.xsq[cf] + 2 * a * b * ed.x[cf] + w * b * b;
		}
		error += ridge_penalty * a * a * feature_variance[cf];
		return error;
	}

	double SimpleLinearRegression::GetTestLeafCosts(const ADataView& data, const BranchContext& context, const LinearModel& model) const {
		return GetLeafCosts(data, context, model);
	}

	void SimpleLinearRegression::GetInstanceLeafD2Costs(const AInstance* i, int org_label, int label, D2SimpleLinRegSol& costs, int multiplier) const {
		auto instance = static_cast<const Instance<double, SimpleLinRegExtraData>*>(i);
		
		auto& ed = instance->GetExtraData();
		costs.weight = multiplier * int(instance->GetWeight());
		costs.ys = multiplier * instance->GetLabel();
		costs.yys = multiplier * ed.ysq;
		if (multiplier == 1) {
			costs.xs = ed.x;
			costs.xsq = ed.xsq;
			costs.yx = ed.yx;
		} else {
			costs.xs.resize(num_cont_features);
			costs.xsq.resize(num_cont_features);
			costs.yx.resize(num_cont_features);
			for (int f = 0; f < num_cont_features; f++) {
				costs.xs[f]  = ed.x[f]   * multiplier;
				costs.xsq[f] = ed.xsq[f] * multiplier;
				costs.yx[f]  = ed.yx[f]  * multiplier;
			}
		}
	}

	inline void GetSimpleModel(const D2SimpleLinRegSol& d2costs, int f, double ridge_penalty, double& a, double& b, double& error) {
		// y = a x + b
		//double var_x = d2costs.xsq[f] / d2costs.weight - (d2costs.xs[f] * d2costs.xs[f] / (d2costs.weight * d2costs.weight));
		//const double& xsq = d2costs.xsq[f];
		//const double& xs = d2costs.xs[f];
		//const double& yx = d2costs.yx[f];
		
		double divisor = (d2costs.weight * d2costs.xsq[f] - d2costs.xs[f] * d2costs.xs[f] + d2costs.weight * ridge_penalty);
		if (std::abs(divisor) < 1e-3) {
			a = 0;
			b = d2costs.ys / d2costs.weight;
			error = d2costs.yys - (d2costs.ys * d2costs.ys / d2costs.weight);
		} else {
			a = (d2costs.weight * d2costs.yx[f] - d2costs.xs[f] * d2costs.ys) / divisor;
			b = (d2costs.ys - a * d2costs.xs[f]) / d2costs.weight;
			error = d2costs.yys - 2 * a * d2costs.yx[f] - 2 * b * d2costs.ys
				+ a * a * d2costs.xsq[f] + 2 * a * b * d2costs.xs[f] + d2costs.weight * b * b
				+ ridge_penalty * a * a;
		}
	}

	void SimpleLinearRegression::ComputeD2Costs(const D2SimpleLinRegSol& d2costs, int count, double& costs) const {
		if (d2costs.weight < minimum_leaf_node_size) {
			costs = DBL_MAX;
			return;
		}
		runtime_assert(d2costs.xs.size() == num_cont_features);
		runtime_assert(d2costs.xsq.size() == num_cont_features);
		runtime_assert(d2costs.yx.size() == num_cont_features);
		costs = DBL_MAX;
		double a, b, error{ 0 };
		int f = num_cont_features;
		while (f--) {
			GetSimpleModel(d2costs, f, ridge_penalty * feature_variance[f], a, b, error);
			if (error < costs) {
				costs = std::max(0.0, error);
			}
		}
	}

	LinearModel SimpleLinearRegression::GetLabel(const D2SimpleLinRegSol& d2costs, int count) const {
		if (d2costs.weight < minimum_leaf_node_size) return LinearModel();
		
		double best_costs = DBL_MAX;
		double a, b, error;
		double best_a = 0;
		double best_b = 0;
		int best_f = 0;
		for (int f = 0; f < num_cont_features; f++) {
			GetSimpleModel(d2costs, f, ridge_penalty * feature_variance[f], a, b, error);
			if (error < best_costs) {
				best_costs = error;
				best_a = a;
				best_b = b;
				best_f = f;
			}
		}
		return LinearModel(num_cont_features, best_f, best_a, best_b);
	}

	PairWorstCount<SimpleLinearRegression> SimpleLinearRegression::ComputeSimilarityLowerBound(const ADataView& data_old, const ADataView& data_new) const {
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
					worst_diff += weight_old * GetWorstPerLabel(GetInstanceLabel<double>(old_instances[index_old]) / weight_old);
					index_old++;
				} else {//no difference
					index_new ++;
					index_old ++;
				}
			}

			
			for (; index_new < size_new; index_new++) {
				int weight_new = int(new_instances[index_new]->GetWeight());
				total_diff += weight_new;
			}
			for (; index_old < size_old; index_old++) {
				int weight_old = int(old_instances[index_old]->GetWeight());
				total_diff += weight_old;
				worst_diff += weight_old * GetWorstPerLabel(GetInstanceLabel<double>(old_instances[index_old]) / weight_old);
			}
		}
		PairWorstCount<SimpleLinearRegression> result(worst_diff, total_diff);
		return result;
	}

	TuneRunConfiguration SimpleLinearRegression::GetTuneRunConfiguration(const ParameterHandler& default_config, const ADataView& data, int tune_phase) {
		TuneRunConfiguration config;

		std::vector<double> alphas = { 0.1, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.0005, 0.0001 };
		std::vector<double> lasso_penalizations = { 1000, 100, 10, 1, 0.1, 0.01, 0 };

		if (tune_phase == 0) {
			ParameterHandler d0_config = default_config;
			d0_config.SetFloatParameter("cost-complexity", .1);
			for (auto lp : lasso_penalizations) {
				ParameterHandler params = d0_config;
				params.SetFloatParameter("lasso-penalty", lp);
				config.AddConfiguration(params, "d0, lp = " + std::to_string(lp));
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
				ParameterHandler params = default_config;
				params.SetFloatParameter("lasso-penalty", lp);
				config.AddConfiguration(params, "lp = " + std::to_string(lp));
			}
			config.skip_when_max_tree = false;
		}
		config.reset_solver = true;
		config.runs = 5;
		return config;
	}

}