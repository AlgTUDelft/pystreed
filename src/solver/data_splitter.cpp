#include "solver/data_splitter.h"

namespace STreeD {

	void DataSplitter::Split(const ADataView& data, const Branch& branch, int feature, ADataView& left, ADataView& right, bool test) {
		if (enabled) {
			auto& hashmap = test ? test_cache[branch.Depth()] : train_cache[branch.Depth()];
			auto iter = hashmap.find({ branch, feature });

			if (iter == hashmap.end()) {
				data.SplitData(feature, left, right);
				hashmap.insert({ {branch, feature}, {left, right} });
				return;
			}
			left = iter->second.first;
			right = iter->second.second;
		} else {
			data.SplitData(feature, left, right);
		}
	}

	void DataSplitter::Clear(bool test) {
		if (test) {
			for (size_t i = 0; i < test_cache.size(); i++)
				test_cache[i].clear();
		} else {
			for (size_t i = 0; i < train_cache.size(); i++)
				train_cache[i].clear();
		}
	}

}