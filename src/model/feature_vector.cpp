/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#pragma warning( disable: 6386 )
#include "model/feature_vector.h"


namespace STreeD {

	FeatureVector::FeatureVector(const std::vector<bool>& feature_values, int id) :
		id(id), num_features(int(feature_values.size())) {
		is_feature_present_ = new bool[feature_values.size()];
		num_present_features = 0;
		for (int feature_index = 0; feature_index < int(feature_values.size()); feature_index++) 
			if (feature_values[feature_index]) num_present_features++;
		present_features_ = new int[num_features];
		int present_feature_ix = 0;
		for (int feature_index = 0; feature_index < int(feature_values.size()); feature_index++) {
			if (feature_values[feature_index]) { present_features_[present_feature_ix++] = feature_index; }
			is_feature_present_[feature_index] = bool(feature_values[feature_index]);
		}
	}

	FeatureVector::FeatureVector(const FeatureVector& fv) 
		: id(fv.id), num_features(fv.num_features), num_present_features(fv.num_present_features) {
		is_feature_present_ = new bool[fv.num_features];
		int num_present_features = 0;
		for (int i = 0; i < fv.num_features; i++) {
			is_feature_present_[i] = fv.is_feature_present_[i];
		}
		present_features_ = new int[num_features];
		for (int i = 0; i < fv.NumPresentFeatures(); i++) {
			present_features_[i] = fv.present_features_[i];
		}
	}

	FeatureVector::~FeatureVector() {
		delete[] is_feature_present_;
		delete[] present_features_;
	}

	double FeatureVector::Sparsity() const {
		return double(NumPresentFeatures()) / NumTotalFeatures();
	}

	void FeatureVector::FlipFeature(int f) {
		if (IsFeaturePresent(f)) {
			std::remove(present_features_, present_features_ + num_present_features, f);
			num_present_features -= 1;
		} else {
			int i;
			for (i = num_present_features; i > 0; i--) {
				if (present_features_[i - 1] < f) {
					present_features_[i] = f;
					break;
				} else {
					present_features_[i] = present_features_[i - 1];
				}
			}
			if (i == 0) {
				present_features_[0] = f;
			}
			num_present_features += 1;
			
		}
		is_feature_present_[f] = !is_feature_present_[f];

		runtime_assert(num_present_features >= 0);
		runtime_assert(num_present_features <= num_features);

		int prev = -1;
		for(int i=0; i<num_present_features; i++) {
			int cur = present_features_[i];
			runtime_assert(cur <= num_features);
			runtime_assert(prev < cur);
			runtime_assert(is_feature_present_[cur]);
		}
		int count = 0;
		for(int i=0; i<num_features; i++) {
			if(is_feature_present_[i]) count++;
		}
		runtime_assert(count == num_present_features);
	}

	std::ostream& operator<<(std::ostream& os, const FeatureVector& fv) {
		if (fv.NumPresentFeatures() == 0) { std::cout << "[empty]"; } else {
			for (int i = 0; i < fv.NumPresentFeatures(); i++) {
				if (i > 0) os << " ";
				os << fv.GetJthPresentFeature(i);
			}
		}
		return os;
	}

}