#include "tasks/accuracy/cost_complex_accuracy.h"

namespace STreeD {

	void CostComplexAccuracy::PreprocessData(AData& data, bool train) {
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
			int unique_feature_vector_id = -1;
			auto prev = instances[0];
			for (auto i : instances) {
				auto cca_i = static_cast<Instance<int, CCAccExtraData>*>(i);
				if (id == 0 || !prev->GetFeatures().HasEqualFeatures(i->GetFeatures())) {
					unique_feature_vector_id++;
					prev = i;
				}
				cca_i->GetMutableExtraData().unique_feature_vector_id = unique_feature_vector_id;
				i->SetID(id++);
			}
		}
	}

	void CostComplexAccuracy::PreprocessTrainData(ADataView& train_data) {
		for (int k = 0; k < train_data.NumLabels(); k++) {
			auto& instances = train_data.GetMutableInstancesForLabel(k);
			std::sort(instances.begin(), instances.end(), [](const AInstance* a, const AInstance* b) {
				return a->GetID() < b->GetID();
			});
		}
	}
	
	double CostComplexAccuracy::GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const { // Replace by custom function later
		double error = 0;
		for (int k = 0; k < data.NumLabels(); k++) {
			if (k == label) continue;
			error += data.NumInstancesForLabel(k);
		}
		return error;
	}

	Node<CostComplexAccuracy> CostComplexAccuracy::ComputeLowerBound(const ADataView& data, const Branch& branch, int max_depth, int num_nodes) {
		// Equivalent Points Bound adapted from Angelino, E., Larus-Stone, N., Alabi, D., Seltzer, M., & Rudin, C. (2018). 
		// Learning certifiably optimal rule lists for categorical data. Journal of Machine Learning Research, 18(234), 1-78.
		auto lb = Node<CostComplexAccuracy>(best);
		auto& hashmap = lower_bound_cache[branch.Depth()];
		auto it = hashmap.find(branch);
		if (it != hashmap.end()) {
			return hashmap[branch];
		}
	
		lb.solution = 0;

		const int num_labels = data.NumLabels();

		std::vector<std::vector<const AInstance*>::const_iterator> iterators;
		std::vector<std::vector<const AInstance*>::const_iterator> ends;
		for (int k = 0; k < num_labels; k++) {
			iterators.push_back(data.GetInstancesForLabel(k).begin());
			ends.push_back(data.GetInstancesForLabel(k).end());
		}

		std::vector<int> labels(num_labels);
		std::iota(labels.begin(), labels.end(), 0);

		auto comp_with_check_end = [&iterators, &ends](const int k1, const int k2) {
			if (iterators[k1] == ends[k1]) return false;
			if (iterators[k2] == ends[k2]) return true;
			return (*iterators[k1])->GetID() < (*iterators[k2])->GetID();
		};
		auto comp = [&iterators, &ends](const int k1, const int k2) {
			return (*iterators[k1])->GetID() < (*iterators[k2])->GetID();
		};
		std::sort(labels.begin(), labels.end(), comp_with_check_end);
		
		// pop empty labels 
		while (iterators[labels[labels.size() - 1]] == ends[labels[labels.size() - 1]])
			labels.pop_back();

		
		// Initialize prev with the first instance in order of the features
		std::vector<int> class_counts(num_labels, 0);
		int current_label = labels[0];
		const AInstance* prev = *iterators[current_label];
		class_counts[current_label]++;
		iterators[current_label]++;
		int n = 1;

		// while the first iterator still has instances
		while (labels.size() > 0 && iterators[labels[0]] != ends[labels[0]]) {

			current_label = labels[0];
			if (static_cast<const Instance<int, CCAccExtraData>*>(prev)->GetExtraData().unique_feature_vector_id ==
				static_cast<const Instance<int, CCAccExtraData>*>(*iterators[current_label])->GetExtraData().unique_feature_vector_id) {
				class_counts[current_label]++;
				n++;
			} else {
				if (n > 1) {
					// Check if the label count is unique
					int largest = class_counts[0];
					for (int k = 1; k < num_labels; k++) {
						if (class_counts[k] > largest) {
							largest = class_counts[k];
						}
					}
					int min_error = n - largest;
					lb.solution += min_error;
				}
				std::fill(class_counts.begin(), class_counts.end(), 0);
				class_counts[current_label]++;
				n = 1;
			}
			prev = *iterators[current_label];
			iterators[current_label]++;
			if (iterators[current_label] == ends[current_label]) {
				labels.erase(labels.begin());
				continue;
			}
			//std::sort(labels.begin(), labels.end(), comp);
			for (int k = 1; k < labels.size(); k++) {
				if (comp(labels[k], current_label)) {
					std::swap(labels[k], labels[k-1]);
				} else {
					break;
				}
			}			
		}
		
		// Compute the error from the last group
		if (n > 1) {
			// Check if the label count is unique
			int largest = class_counts[0];
			for (int k = 1; k < num_labels; k++) {
				if (class_counts[k] > largest) {
					largest = class_counts[k];
				}
			}
			int min_error = n - largest;
			lb.solution += min_error;
		}
		hashmap[branch] = lb;
		return lb;
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
		std::sort(alphas.begin(), alphas.end(), std::greater<>());
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