/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#pragma once
#include "base.h"

namespace STreeD {

	class FeatureVector {
	public:
		FeatureVector(const std::vector<bool>& feature_values, int id);
		FeatureVector(const FeatureVector& fv);
		~FeatureVector();

		inline bool IsFeaturePresent(const int feature) const { return is_feature_present_[feature]; }
		inline int GetJthPresentFeature(const int j) const { runtime_assert(j < num_present_features); return present_features_[j]; }
		inline int NumPresentFeatures() const { return int(num_present_features); }
		inline int NumTotalFeatures() const { return num_features; }
		inline int NumPresentFeaturePairs() const { return int(present_feature_pair_indices_.size()); }
		inline int GetID() const { return id; }
		inline void SetID(int id) { this->id = id; }
		double Sparsity() const;
		inline int GetJthPresentFeaturePairIndex(const int j) const { return present_feature_pair_indices_[j]; }
		inline const std::vector<int>& GetPresentFeaturePairIndices() const { return present_feature_pair_indices_; }
		void FlipFeature(int f);
		void DisableFeature(int f);
		void ComputeFeaturePairIndices();

		friend std::ostream& operator<<(std::ostream& os, const FeatureVector& fv);
		
		inline bool HasEqualFeatures(const FeatureVector& v2) const {
			if (num_present_features != v2.num_present_features) return false;
			for (int i = 0; i < num_present_features; i++) {
				if (present_features_[i] != v2.present_features_[i]) {
					return false;
				}
			}
			return true;
		}

		inline bool operator>(const FeatureVector& v2) const {
			for (int i = 0; i < num_features; i++) {
				bool present = is_feature_present_[i];
				bool present2 = v2.is_feature_present_[i];
				if (present && !present2) {
					return true;
				}
				else if (!present && present2) {
					return false;
				}
			}

			return false;
		}

	private:
		int id;
		int num_features;
		int num_present_features;
		bool* is_feature_present_; //[i] indicates if the feature is true or false, i.e., if it is present in present_Features.
		int* present_features_;
		std::vector<int> present_feature_pair_indices_;

	};
}