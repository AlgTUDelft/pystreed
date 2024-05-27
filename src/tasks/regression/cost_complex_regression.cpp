#include "tasks/regression/cost_complex_regression.h"
#include "solver/similarity_lowerbound.h"

namespace STreeD {

	double GetYSQ(const AInstance* instance) {
		return static_cast<const Instance<double, RegExtraData>*>(instance)->GetExtraData().ysq;
	}

	void CostComplexRegression::PreprocessData(AData& data, bool train) {
		std::vector<AInstance*>& instances = data.GetInstances();
		int n = int(instances.size());

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

			normalize_scale = 1; // std::sqrt(yy - (y * y / n)); // Scale by RMSE of the dataset
		}

		for (auto i : instances) {
			LInstance<double>* li = static_cast<LInstance<double>*>(i);
			double label = li->GetLabel() / normalize_scale;
			li->SetLabel(label);
			auto reg_i = static_cast<Instance<double, RegExtraData>*>(i);
			reg_i->GetMutableExtraData().ysq = label * label;
		}
	}

	void CostComplexRegression::PreprocessTrainData(ADataView& train_data) {
		auto& instances = train_data.GetMutableInstancesForLabel(0);
				
		std::sort(instances.begin(), instances.end(), [](const AInstance* a, const AInstance* b) {
			return a->GetID() < b->GetID();
		});


		// Compute bounds for the similarity lower bound
		{
			min = GetInstanceLabel<double>(instances[0]) / instances[0]->GetWeight();
			max = min;
			double ys = 0;
			double yys = 0;
			total_training_weight = 0;
			for (auto i : instances) {
				auto instance = static_cast<const Instance<double, RegExtraData>*>(i);
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
			branching_cost = cost_complexity_parameter;
			if (normalize_scale == 1) branching_cost *= cost_complexity_norm_coef; // Multiply by coef when target iff target is not scaled
			runtime_assert(branching_cost >= 0);

			double worst = max - min;
			worst_distance_squared = worst * worst;
		}


		
		auto prev = instances[0];
		int ix_first = 0;
		double y = GetInstanceLabel<double>(prev);
		double ysq = GetInstanceExtraData<double, RegExtraData>(prev).ysq;
		int w = int(prev->GetWeight());
		int n = w;

		//int total_weight = int(instances.size());

		for (int i = 1; i < instances.size(); i++) {
			auto current = instances[i];
			
			double label = GetInstanceLabel<double>(current);
			double delta_ysq = GetInstanceExtraData<double, RegExtraData>(current).ysq;
			

			if (//std::abs(GetInstanceLabel<double>(prev) - label) > 1e-3 ||
				  !prev->GetFeatures().HasEqualFeatures(current->GetFeatures())) {
				// If different
				if (n > w) {
					auto copy_instance = new Instance<double, RegExtraData>(*static_cast<const Instance<double, RegExtraData>*>(instances[ix_first]));
					//auto mutable_instance = train_data.GetMutableInstance(0, ix_first);
					SetInstanceLabel<double>(copy_instance, y); // Set average
					auto reg_i = static_cast<Instance<double, RegExtraData>*>(copy_instance);
					reg_i->GetMutableExtraData().ysq = ysq;
					copy_instance->SetWeight(n);
					instances[ix_first] = copy_instance;
					data.AddInstance(copy_instance);
				} 
				instances[++ix_first] = current; // Swap instances, (to prevent having to do an expensive erase operation)
				y = 0;
				ysq = 0;
				n = 0;
			} 
			w = int(current->GetWeight());
			n += w;
			y += label;
			ysq += delta_ysq;
			prev = current;
		}
		{
			auto mutable_instance = train_data.GetMutableInstance(0, ix_first);
			SetInstanceLabel<double>(mutable_instance, y);
			auto reg_i = static_cast<Instance<double, RegExtraData>*>(mutable_instance);
			reg_i->GetMutableExtraData().ysq = ysq;
			mutable_instance->SetWeight(n);
		}
		instances.resize(ix_first+1);
		train_data.ComputeSize();

		int _check_weight = 0;
		for (int i = 0; i < instances.size(); i++) {
			_check_weight += int(instances[i]->GetWeight());
		}
		runtime_assert(_check_weight == total_training_weight);
	}

	void CostComplexRegression::InformTrainData(const ADataView& train_data, const DataSummary& train_summary) {
		OptimizationTask::InformTrainData(train_data,  train_summary);

		const std::vector<const AInstance*>& instances = train_data.GetInstancesForLabel(0);

		for (auto& map : lower_bound_cache) {
			map.clear();
		}
	}

	void CostComplexRegression::InformTestData(const ADataView& test_data, const DataSummary& test_summary) {
		OptimizationTask::InformTestData(test_data, test_summary);

		// Compute test total variance
		double test_ys = 0;
		double test_yys = 0;
		int test_weight = 0;
		for (auto i : test_data.GetInstancesForLabel(0)) {
			auto instance = static_cast<const Instance<double, RegExtraData>*>(i);
			double label = instance->GetLabel();
			double ysq = label * label;
			int weight = int(i->GetWeight());
			test_ys += label;
			test_yys += ysq;
			test_weight += weight;
		}
		test_total_variance = test_yys - (test_ys * test_ys / test_weight);
	}

	Node<CostComplexRegression> CostComplexRegression::SolveLeafNode(const ADataView& data, const BranchContext& context) const {
		double ysq = 0, y = 0;
		int weight = 0;
		for (const auto i : data.GetInstancesForLabel(0)) {
			auto instance = static_cast<const Instance<double, RegExtraData>*>(i);
			int w = int(i->GetWeight());
			ysq += instance->GetExtraData().ysq;
			y += instance->GetLabel();
			weight += w;
		}
		if (weight < minimum_leaf_node_size) return Node<CostComplexRegression>();
		double label = y / weight;
		double error = std::max(0.0, ysq - (y * y / weight));
		//runtime_assert(error >= -1e-6);
		return Node<CostComplexRegression>(label, error);
	}

	double CostComplexRegression::GetLeafCosts(const ADataView& data, const BranchContext& context, double label) const {
		// Here we cannot assume that the label is equal to the average label
		double ysq = 0, y = 0;
		int weight = 0;
		for (const auto i : data.GetInstancesForLabel(0)) {
			auto instance = static_cast<const Instance<double, RegExtraData>*>(i);
			int w = int(instance->GetWeight());
			ysq += instance->GetExtraData().ysq;
			y += instance->GetLabel();
			weight += w;
		}
		return std::max(0.0, ysq - 2 * label * y + weight * label * label);
	}

	double CostComplexRegression::GetTestLeafCosts(const ADataView& data, const BranchContext& context, double label) const {
		return GetLeafCosts(data, context, label);
	}

	void CostComplexRegression::GetInstanceLeafD2Costs(const AInstance* i, int org_label, int label, D2CostComplexRegressionSol& costs, int multiplier) const {
		auto instance = static_cast<const Instance<double, RegExtraData>*>(i);
		costs.ys = multiplier * instance->GetLabel();
		costs.yys = multiplier * instance->GetExtraData().ysq;
		costs.weight = multiplier * int(instance->GetWeight());
	}

	void CostComplexRegression::ComputeD2Costs(const D2CostComplexRegressionSol& d2costs, int count, double& costs) const {
		if (count == 0) {
			costs = DBL_MAX;
			return;
		}
		costs = d2costs.yys - (d2costs.ys * d2costs.ys / d2costs.weight); // MSE error
		costs = costs < 0 ? 0.0 : costs;
	}

	double CostComplexRegression::GetLabel(const D2CostComplexRegressionSol& costs, int count) const {
		if (count == 0) return 0;
		return costs.ys / costs.weight; // average of sum(y)
	}

	Node<CostComplexRegression> CostComplexRegression::ComputeLowerBound(const ADataView& data, const Branch& branch, int max_depth, int num_nodes) {
		// k-Means Equivalent Points Bound adapted from Optimal Sparse Regression Trees by Zhang et al. https://doi.org/10.48550/arXiv.2211.14980
		auto lb = Node<CostComplexRegression>(best);
		auto& hashmap = lower_bound_cache[branch.Depth()];
		auto& solutions = hashmap[branch];
		const int hard_max_k = use_kmeans ? 50 : 1;
		const int solution_idx = std::min(hard_max_k, num_nodes + 1) - 1;
		const Node<CostComplexRegression> best_node(best);

		if (solutions.size() <= solution_idx) {
			solutions.resize(solution_idx + 1, best_node);
		}
		if (solutions[solution_idx].solution != best) {
			lb = solutions[solution_idx];
			return lb;
		}

		lb.solution = 0;
		
		if (use_kmeans) {
			weight_value_pairs.resize(data.Size());
		}

		const std::vector<const AInstance*>& instances = data.GetInstancesForLabel(0);
		

		// Find groups of instances with equivalent features. Since the instances are sorted by features
		// in the preprocessing step, equivalent featured instances are always consecutive.
		const AInstance* prev = instances[0];
		int w = int(prev->GetWeight());
		double y = GetInstanceLabel<double>(prev);
		double ysq = GetYSQ(prev);
		int n = w;

		int ix = 0;
		const int size = int(instances.size());
		for (int i = 1; i < size; i++) {
			const auto& current = instances[i];
			if (!prev->GetFeatures().HasEqualFeatures(current->GetFeatures())) {

				lb.solution += ysq - (y * y / n);

				if (use_kmeans) {
					weight_value_pairs[ix].value = y / n;
					weight_value_pairs[ix].weight = n;
				}

				n = 0;
				y = 0;
				ysq = 0;
				ix++;
			}

			n += int(current->GetWeight());;
			double label = GetInstanceLabel<double>(current);
			y += label;
			ysq += GetYSQ(current);
			prev = current;
		}
		// Process last group of instances with equivalent features
		lb.solution += ysq - (y * y / n);
		if (num_nodes > 0)
			lb.num_nodes_left = num_nodes - 1;

		if (use_kmeans) {
			weight_value_pairs[ix].value = y / n;
			weight_value_pairs[ix++].weight = n;
			weight_value_pairs.resize(ix);

			int N = int(weight_value_pairs.size());
			if (N != 1) {
				// Max num_nodes branching nodes, means a maximum of num_nodes + 1 leaf nodes.
				int maxKmeans = std::min(num_nodes + 1, N);
				runtime_assert(maxKmeans > 0);

				// If maxKmeans >= hard_max_k then don't use kmeans, which means we can use the cache of other num_nodes that don't use kmeans
				auto& cache_try = solutions[std::min(hard_max_k, maxKmeans) - 1];
				if (cache_try.solution != best) {
					lb = cache_try;
				} else if (maxKmeans < hard_max_k) {

					std::sort(weight_value_pairs.begin(), weight_value_pairs.end(), 
						[&](WeightValuePair p1, WeightValuePair p2) { return p1.value < p2.value; } );

					S.clear();
					J.clear();
					S.resize(maxKmeans, std::vector<double>(N));
					J.resize(maxKmeans, std::vector<size_t>(N));

					// todo return both bound + nodes
					auto bound = fill_dp_matrix_dynamic_stop(weight_value_pairs, S, J, branching_cost);
					lb.solution += bound.first; // TODO: why += ? Why not =
					if (bound.second > 0)
						lb.num_nodes_left = bound.second - 1;
					else lb.num_nodes_left = 0;

					// Also cache lower K if it was lower
					for(int i=bound.second; i < maxKmeans; i++)
						solutions[i] = lb;
					//solutions[maxKmeans - 1] = lb;
				}
			}
		}

		solutions[solution_idx] = lb;

		return lb;
	}

	PairWorstCount<CostComplexRegression> CostComplexRegression::ComputeSimilarityLowerBound(const ADataView& data_old, const ADataView& data_new) const {
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
		PairWorstCount<CostComplexRegression> result(worst_diff, total_diff);
		return result;
	}

	TuneRunConfiguration CostComplexRegression::GetTuneRunConfiguration(const ParameterHandler& default_config, const ADataView& data, int phase) {
		TuneRunConfiguration config;

		std::vector<double> alphas = { 0.1, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.0005, 0.0001 };
		for (auto a : alphas) {
			ParameterHandler params = default_config;
			params.SetFloatParameter("cost-complexity", a);
			config.AddConfiguration(params, "a = " + std::to_string(a));
		}
		config.reset_solver = true;
		config.skip_when_max_tree = true;
		return config;
	}

}