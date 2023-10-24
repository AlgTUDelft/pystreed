#pragma once
#include "base.h"

namespace STreeD {

	/*
	* Class that stores information about the branch
	* Note that the class does _not_ preserve the original order of the branching decisions, but
	* sorts the decisions ascendingly
	* When branching left on feature f, the branching code is 2*f
	* When branching right on feature f, the branching code is 2*f + 1
	*/
	class Branch {
	public:
		Branch() = default;
		Branch(const Branch& other);

		inline int Depth() const { return int(branch_codes.size()); }
		inline int operator[](int i) const { return branch_codes[i]; }

		static Branch LeftChildBranch(const Branch& branch, int feature);
		static Branch RightChildBranch(const Branch& branch, int feature);

		static void LeftChildBranch(const Branch& branch, int feature, Branch& out);
		static void RightChildBranch(const Branch& branch, int feature, Branch& out);

		bool HasBranchedOnFeature(int feature) const;

		bool operator==(const Branch& right_hand_side) const;
		inline bool operator!=(const Branch& right_hand_side) const { return !(*this == right_hand_side); }
		
		friend std::ostream& operator<<(std::ostream& out, const Branch& branch) {
			for (int code : branch.branch_codes) {
				out << code << " ";
			}
			return out;
		}

	private:
		inline int GetCode(int feature, bool present) const { return 2 * feature + present; }
		void AddFeatureBranch(int feature, bool present);
		void ConvertIntoCanonicalRepresentation();

		std::vector<int> branch_codes;
	};

	//adapted from https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
	//see also https://stackoverflow.com/questions/4948780/magic-number-in-boosthash-combine
	//and https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
	struct BranchHashFunction {
		//todo check about overflows
		int operator()(Branch const& branch) const {
			int seed = int(branch.Depth());
			for (int i = 0; i < branch.Depth(); i++) {
				int code = branch[i];
				seed ^= code + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			}
			return seed;
		}
	};

	//assumes that both inputs are in canonical representation
	struct BranchEquality {
		bool operator()(Branch const& branch1, Branch const& branch2) const {
			if (branch1.Depth() != branch2.Depth()) { return false; }
			for (int i = 0; i < branch1.Depth(); i++) {
				if (branch1[i] != branch2[i]) { return false; }
			}
			return true;
		}
	};
}