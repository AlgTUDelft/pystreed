#pragma once
#include "base.h"
#include "model/data.h"
#include "model/branch.h"
#include "solver/branch_cache.h"

namespace STreeD {

	//adapted from https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
	//see also https://stackoverflow.com/questions/4948780/magic-number-in-boosthash-combine
	//and https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
	struct BranchFeatureHashFunction {
		//todo check about overflows
		int operator()(const std::pair<Branch, int>& o) const {
			int seed = BranchHashFunction()(o.first);
			seed ^= o.second + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			return seed;
		}
	};

	//assumes that both inputs are in canonical representation
	struct BranchFeatureEquality {
		bool operator()(const std::pair<Branch, int>& o1, const std::pair<Branch, int>& o2) const {
			return (o1.first == o2.first && o1.second == o2.second);
		}
	};

	// Class for storing historic data splits, to prevent recomputing similar data splits
	class DataSplitter {
	public:
		using Key = std::pair<Branch, int>;
		using Value = std::pair<ADataView, ADataView>;

		DataSplitter(int max_branch_length) : train_cache(max_branch_length), test_cache(max_branch_length) {}

		void Split(const ADataView& data, const Branch& branch, int feature, ADataView& left, ADataView& right, bool test=false);
		void Clear(bool test = false);
		void Disable() { enabled = false; }

	private:
		std::vector<std::unordered_map<Key, Value, BranchFeatureHashFunction, BranchFeatureEquality>> train_cache, test_cache;
		bool enabled{ true };
	};

}