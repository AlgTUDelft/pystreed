/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#include "solver/feature_selector.h"

namespace STreeD {

	void FeatureSelectorGini::InitializeInternal(const ADataView& data) {

		uint32_t data_size = data.Size();

		//clear helper data structures
		double max_gini_value = -1.0;
		const int num_labels = data.NumLabels();

		std::vector<std::vector<int> > num_label_with_feature(num_labels, std::vector<int>(num_features, 0));
		std::vector<std::vector<int> > num_label_without_this_feature(num_labels, std::vector<int>(num_features, 0));
		std::vector<int> num_with_feature(num_features, 0);
		std::vector<int> num_without_feature(num_features, 0);
		std::vector<double> gini_values(num_features, 0);

		for (int label = 0; label < data.NumLabels(); label++) {
			for (auto& instance : data.GetInstancesForLabel(label)) {
				for (int feature = 0; feature < num_features; feature++) {
					if (instance->IsFeaturePresent(feature)) {
						num_label_with_feature[label][feature]++;
						num_with_feature[feature]++;
					} else {
						num_label_without_this_feature[label][feature]++;
						num_without_feature[feature]++;
					}
				}
			}
		}

		//compute the gini values for each feature 
		double I_D = 1.0;
		for (int label = 0; label < num_labels; label++) {
			I_D -= pow(double(data.NumInstancesForLabel(label)) / data.Size(), 2);
		}
		
		for (int feature = 0; feature < num_features; feature++) {

			double I_D_without_feature = 1.0;
			if (num_without_feature[feature] > 0) {
				for (int label = 0; label < num_labels; label++) {
					I_D_without_feature -= pow(double(num_label_without_this_feature[label][feature]) / num_without_feature[feature], 2);
				}
			}

			double I_D_with_feature = 1.0;
			if (num_with_feature[feature] > 0) {
				for (int label = 0; label < num_labels; label++) {
					I_D_with_feature -= pow(double(num_label_with_feature[label][feature]) / num_with_feature[feature], 2);
				}
			}

			gini_values[feature] = I_D - (double(num_without_feature[feature]) / data_size) * I_D_without_feature
				- (double(num_with_feature[feature]) / data_size) * I_D_with_feature;
			
			max_gini_value = std::max(gini_values[feature], max_gini_value);

		}

		while (feature_order.Size() != 0) { feature_order.PopMax(); }
		for (int feature = 0; feature < num_features; feature++) {
			feature_order.Readd(feature);
			double heap_value = gini_values[feature];
			//if (ascending_) { heap_value = max_gini_value_ - heap_value; } //converting a max heap into a min heap
			feature_order.Increment(feature, heap_value);
		}

	}
}