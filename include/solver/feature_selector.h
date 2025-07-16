#pragma once
#include "base.h"
#include "model/data.h"
#include "utils/key_value_heap.h"

namespace STreeD {

	class FeatureSelectorAbstract {
	public:
		FeatureSelectorAbstract() = delete;
		FeatureSelectorAbstract(int num_features) : num_features(num_features), num_features_popped(0) {}
		virtual ~FeatureSelectorAbstract() {}
		
		void Initialize(const ADataView& data) {
			InitializeInternal(data);
		}

		int PopNextFeature() {
			runtime_assert(AreThereAnyFeaturesLeft());
			int next_feature = PopNextFeatureInternal();
			num_features_popped++;
			return next_feature;
		}
		inline bool AreThereAnyFeaturesLeft() const { return num_features_popped != num_features;  }
		void PopAllFeatures() {
			while (AreThereAnyFeaturesLeft()) PopNextFeature();
		}

	protected:
		virtual int PopNextFeatureInternal() = 0;
		
		virtual void InitializeInternal(const ADataView& data) = 0;

		int num_features;
		int num_features_popped;
	};

	class FeatureSelectorInOrder : public FeatureSelectorAbstract {
	public:
		FeatureSelectorInOrder() = delete;
		FeatureSelectorInOrder(int num_features) : FeatureSelectorAbstract(num_features), next(0) {}
		~FeatureSelectorInOrder() = default;
	protected:
		int PopNextFeatureInternal() { return next++;  }
		
		void InitializeInternal(const ADataView& data) {}
	private:
		int next;
	};

	class FeatureSelectorGini : public FeatureSelectorAbstract {
	public:
		FeatureSelectorGini() = delete;
		FeatureSelectorGini(int num_features) : FeatureSelectorAbstract(num_features), feature_order(num_features) {}
		~FeatureSelectorGini() = default;
	protected:
		int PopNextFeatureInternal() {
			return feature_order.PopMax();
		}
		void InitializeInternal(const ADataView& data);

	private:
		KeyValueHeap feature_order;
	};

	class FeatureSelectorMSE : public FeatureSelectorAbstract {
	public:
		FeatureSelectorMSE() = delete;
		FeatureSelectorMSE(int num_features) : FeatureSelectorAbstract(num_features), feature_order(num_features) {}
		~FeatureSelectorMSE() = default;
	protected:
		int PopNextFeatureInternal() {
			return feature_order.PopMax();
		}
		void InitializeInternal(const ADataView& data);

	private:
		KeyValueHeap feature_order;
	};
}