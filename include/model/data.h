#pragma warning( disable: 26451 )
#pragma once
#include "base.h"
#include "model/feature_vector.h"
#include <functional>
#include "utils/dynamic_bitset.h"

namespace STreeD {

	// Extra data class for optimization tasks that have more data than only a label and feature data
	class ExtraData {
	public:
		static ExtraData ReadData(std::istringstream& iss, int num_labels) { return ExtraData(); }
	};

	// Instance class for storing the feature vector
	class AInstance {
	public:
		AInstance() = delete;
		AInstance(int id, const std::vector<bool>& feature_vector) : id(id), features(feature_vector, id), weight(1.0) {}

		// The unique identifier of this instance
		inline const int GetID() const { return id; }
		
		// Change the unique identifier of this instance. This function should rarely be used
		inline void SetID(int new_id) { id = new_id; features.SetID(id); }
		
		// Get the feature vector of this instance
		inline const FeatureVector& GetFeatures() const { return features; }

		// Check if this instance satisfies a feature
		inline bool IsFeaturePresent(int feature) const { return features.IsFeaturePresent(feature); }
		
		// Return the total number of features that are satisfied (present) in this instance
		inline int NumPresentFeatures() const { return features.NumPresentFeatures(); }
		
		// Get the j-th feature that is satisfied in this instance
		inline int GetJthPresentFeature(int j) const { return features.GetJthPresentFeature(j); }

		// Get the j-th feature pair index of two features that are both satisfied in this instance
		inline int GetJthPresentFeaturePairIndex(int j) const { return features.GetJthPresentFeaturePairIndex(j); }

		// Get the number of feature pair indices that are both satisfied in this instance
		inline int NumPresentFeaturePairs() const { return features.NumPresentFeaturePairs(); }

		// Get a list of feature pair indices of two features that are both satisfied in this instance
		inline const std::vector<int>& GetPresentFeaturePairIndices() const { return features.GetPresentFeaturePairIndices(); }

		// Flip the value of feature f. Add or remove this feature from the present-feature list
		inline void FlipFeature(int f) { features.FlipFeature(f); }

		// Disable this feature by setting it to zero. Remove it from the present-feature list
		inline void DisableFeature(int f) { features.DisableFeature(f); }

		// Compute the feature pair indices of the feature pairs that are both satisfied in this instance
		inline void ComputeFeaturePairIndices() { features.ComputeFeaturePairIndices(); }
		
		// Get the weigth of this instance (default = 1)
		// Note: only some optimization tasks support the use of weight
		inline double GetWeight() const { return weight; }

		// Set the weight of this instance
		// Note: only some optimization tasks support the use of weight
		inline void SetWeight(double weight) { this->weight = weight; }
	protected:
		int id;
		double weight;
		FeatureVector features;
	};

	// Instance class that contains both the feature vector and the label
	template <class LT>
	class LInstance : public AInstance {
	public:
		LInstance() = delete;
		LInstance(int id, const std::vector<bool>& feature_vector, LT label)
			: AInstance(id, feature_vector), label(label) { }

		// Get the label of the instance
		inline const LT GetLabel() const { return label; }

		// Set the label of the instance
		inline void SetLabel(LT new_label) { label = new_label; }
	protected:
		LT label{};
	};

	// Instance class that contains the feature vector and the label, and the extra data
	template <class LT, class ET>
	class Instance : public LInstance<LT> {
	public:
		Instance() = delete;
		Instance(int id, const std::vector<bool>& feature_vector, LT label, ET extra_data = ET())
			: LInstance<LT>(id, feature_vector, label), extra_data(extra_data) {}

		// Get the extra data (data beyond the feature vector and label of the instance
		inline const ET& GetExtraData() const { return extra_data; }

		// Set the extra data (data beyond the feature vector and label of the instance
		inline ET& GetMutableExtraData() { return extra_data; }

	protected:
		ET extra_data{};
	};

	// Convenience function that returns the label of an instance
	template<class LT>
	inline LT GetInstanceLabel(const AInstance* instance) {
		return static_cast<const LInstance<LT>*>(instance)->GetLabel();
	}

	// Convenience function that sets the label of an instance
	template<class LT>
	inline void SetInstanceLabel(AInstance* instance, LT label) {
		static_cast<LInstance<LT>*>(instance)->SetLabel(label);
	}

	// Convenience function that returns the extra data of an instance
	template<class LT, class ET>
	inline const ET& GetInstanceExtraData(const AInstance* instance) {
		return static_cast<const Instance<LT, ET>*>(instance)->GetExtraData();
	}

	// The data class which stores all the instances
	// Instances are stored in vectors for every unique label
	// Except for tasks with a double label. In that case, all instances are stored in index 0
	class AData {
	public:
		AData() : num_features(INT32_MAX) {}
		AData (int num_features) : num_features(num_features){}
		~AData() {
			Clear();
		}
		inline void Clear() {
			for (auto instance : instances) delete instance;
			instances.clear();
		}

		inline void AddInstance(AInstance* instance) { instances.push_back(instance); }
		inline void SetNumFeatures(int num_features) { this->num_features = num_features; }

		inline int Size() const { return int(instances.size()); }
		inline int NumFeatures() const { return num_features; }
		inline const AInstance* GetInstance(int ix) const { return instances[ix]; }
		inline AInstance* GetMutableInstance(int ix) const { return instances[ix]; }

		// Get mutable vector of instances, used for example when preprocessing the data for a task.
		// It is the responsibility of the caller that the resulting data is still valid.
		// E.g. the ID's of the instances should be sequential, the number of features should be
		// consistent with num_features, etc.
		inline std::vector<AInstance*>& GetInstances() { return instances; }

	protected:
		std::vector<AInstance*> instances;
		int num_features;
	};

	class ADataView;

	// A bitset representation of the data
	// Every i-th bit represents if the i-th instance is present or not
	struct ADataViewBitSet {
		DynamicBitSet bitset;
		size_t size{ 0 };
		long long hash{ -1 };

		ADataViewBitSet() = default;
		ADataViewBitSet(const ADataView& data);
		inline bool operator==(const ADataViewBitSet& other) const { return size == other.size && bitset == other.bitset; }
		inline bool operator!=(const ADataViewBitSet& other) const { return !((*this) == other); }
		inline size_t Size() const { return size; }
		inline bool IsHashSet() const { return hash != -1; }
		inline bool IsBitViewSet() const { return size > 0; }
		inline void SetHash(long long _hash) { hash = _hash; }
		inline long long GetHash() const { runtime_assert(IsHashSet()); return hash;  }
	};


	// A Dataview of the Data
	// Contains a vector of pointers to the instances
	class ADataView {
	public:
		ADataView() : ADataView(nullptr, {}) {}
		ADataView(const AData* data, int num_labels) : data(data), size(0) { instances.resize(num_labels); instance_weights.resize(num_labels); }
		ADataView(const AData* data, const std::vector<std::vector<const AInstance*>>& instances, const std::vector<std::vector<double>>& instance_weights = {});
		
		inline const std::vector<const AInstance*>& GetInstancesForLabel(int label) const { return instances[label];  }
		inline std::vector<const AInstance*>& GetMutableInstancesForLabel(int label)  { return instances[label]; }
		inline std::vector<std::vector<const AInstance*>>& GetMutableInstances() { return instances; }
		inline int NumInstancesForLabel(int label) const { return int(instances[label].size()); }
		inline AInstance* GetMutableInstance(int label, int ix) const { return data->GetMutableInstance(instances[label][ix]->GetID()); }
		inline int NumLabels() const { return int(instances.size()); }
		inline int NumFeatures() const { return data->NumFeatures(); }
		inline int Size() const { return size; }
		void ComputeSize();
		inline const AData* GetData() const { return data; }
		inline bool IsInitialized() const { return data != nullptr; }
		inline ADataViewBitSet& GetBitSetView() {
			if (!bitset_view.IsBitViewSet()) bitset_view = ADataViewBitSet(*this);
			return bitset_view;
		}

		// Hashing functions, for use in the dataset cache
		inline long long GetHash() const {	return bitset_view.GetHash(); }
		inline bool IsHashSet() const { return bitset_view.IsHashSet(); }
		inline void  SetHash(long long hash) { bitset_view.SetHash(hash); }

		inline void Initialize(const AData* data, int num_labels);
		inline void Clear() { instances.clear(); instance_weights.clear(); size = 0; bitset_view = ADataViewBitSet(); }
		void ResetReserve(const ADataView& other);

		// Split the data on feature
		void SplitData(int feature, ADataView& left, ADataView& right) const;
		
		template<class LT>
		static void TrainTestSplitData(const ADataView& all_data, ADataView& train, ADataView& test, std::default_random_engine* rng, double test_percentage, bool stratify);

		template<class LT>
		static void KFoldSplit(const ADataView& all_data, std::vector<ADataView>& train, std::vector<ADataView>& test, std::default_random_engine* rng, int folds, bool stratify);

		void AddInstance(int label, const AInstance* instance);

		bool operator==(const ADataView& other) const;

	protected:
		std::vector<std::vector<const AInstance*>> instances; // vector of instances per label (if regression, all in the first)
		std::vector<std::vector<double>> instance_weights;
		ADataViewBitSet bitset_view;
		const AData* data;
		int size;
	};

	// A summary of the data
	struct DataSummary {
		DataSummary() : size(0), num_features(0), num_labels(0) {}
		DataSummary(const ADataView& data) :
			size(data.Size()), num_features(data.NumFeatures()), num_labels(data.NumLabels())
		{
			for (int i = 0; i < data.NumLabels(); i++) {
				instances_per_class.push_back(data.NumInstancesForLabel(i));
			}
		}
		int size, num_features, num_labels;
		std::vector<int> instances_per_class;
	};


}

namespace std {

	template <>
	struct hash<STreeD::ADataViewBitSet> {

		size_t operator()(const STreeD::ADataViewBitSet& view) const {
			if (view.IsHashSet()) return view.GetHash();
			return hash<STreeD::DynamicBitSet>()(view.bitset);
		}

	};
}