/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#include "solver/feature_selector.h"

namespace STreeD {

	void FeatureSelectorMSE::InitializeInternal(const ADataView& data) {
		// This feature selector only works for regression labels (i.e., numlabels = 1)
		runtime_assert(data.NumLabels() == 1);
		
		size_t data_size = data.Size();

		//clear helper data structures
		double max_mse_value = -1.0;
		std::vector<double> sum_of_y_with_feature(num_features, 0);
		std::vector<double> sum_of_ysq_with_feature(num_features, 0);
		std::vector<double> count_with_feature(num_features, 0);
		std::vector<double> sum_of_y_without_feature(num_features, 0);
		std::vector<double> sum_of_ysq_without_feature(num_features, 0);
		std::vector<double> count_without_feature(num_features, 0);
		std::vector<double> mse_values(num_features, 0);
		
		for (auto& instance : data.GetInstancesForLabel(0)) {
			double count = instance->GetWeight();
			double sum_y = GetInstanceLabel<double>(instance); // Because of preprocessing, y stores the sum of y
			double sum_ysq = (sum_y / count) * sum_y; // (sum_y/count)^2
			for (int feature = 0; feature < num_features; feature++) {
				if (instance->IsFeaturePresent(feature)) {
					sum_of_y_with_feature[feature] += sum_y;
					sum_of_ysq_with_feature[feature] += sum_ysq;
					count_with_feature[feature] += count;
				} else {
					sum_of_y_without_feature[feature] += sum_y;
					sum_of_ysq_without_feature[feature] += sum_ysq;
					count_without_feature[feature] += count;
				}
			}
		}
		

		//compute the mse values for each feature 
		
		for (int feature = 0; feature < num_features; feature++) {

			double SSE_without_feature = 0.0;
			if (count_without_feature[feature] > 0) {
				SSE_without_feature = sum_of_ysq_without_feature[feature] 
					- sum_of_y_without_feature[feature] * sum_of_y_without_feature[feature] / count_without_feature[feature];
			}

			double SSE_with_feature = 0.0;
			if (count_with_feature[feature] > 0) {
				SSE_with_feature = sum_of_ysq_with_feature[feature]
					- sum_of_y_with_feature[feature] * sum_of_y_with_feature[feature] / count_with_feature[feature];
			}

			mse_values[feature] = (SSE_without_feature + SSE_with_feature) / data_size;
			
			max_mse_value = std::max(mse_values[feature], max_mse_value);

		}

		while (feature_order.Size() != 0) { feature_order.PopMax(); }
		const bool ascending_ = true;
		for (int feature = 0; feature < num_features; feature++) {
			feature_order.Readd(feature);
			double heap_value = mse_values[feature];
			if (ascending_) { heap_value = max_mse_value - heap_value; } //converting a max heap into a min heap
			feature_order.Increment(feature, heap_value);
		}

	}
}