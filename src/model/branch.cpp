/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#include "model/branch.h"

namespace STreeD {

	Branch::Branch(const Branch& other) : branch_codes(other.branch_codes) {}

	
	void Branch::AddFeatureBranch(int feature, bool present) {
		int code = GetCode(feature, present);
		branch_codes.push_back(code);

		ConvertIntoCanonicalRepresentation();
	}

	
	Branch Branch::LeftChildBranch(const Branch& branch, int feature) {
		Branch left_child_branch(branch);
		left_child_branch.AddFeatureBranch(feature, false); //the convention is that the left branch does not have the feature
		return left_child_branch;
	}

	
	Branch Branch::RightChildBranch(const Branch& branch, int feature) {
		Branch right_child_branch(branch);
		right_child_branch.AddFeatureBranch(feature, true); //the convention is that the right branch has the feature
		return right_child_branch;
	}

	
	void Branch::LeftChildBranch(const Branch& branch, int feature, Branch& out) {
		out.branch_codes = branch.branch_codes;
		out.AddFeatureBranch(feature, false);
	}

	
	void Branch::RightChildBranch(const Branch& branch, int feature, Branch& out) {
		out.branch_codes = branch.branch_codes;
		out.AddFeatureBranch(feature, true);
	}

	
	bool Branch::operator==(const Branch& right_hand_side) const {
		if (this->branch_codes.size() != right_hand_side.branch_codes.size()) { return false; }
		for (size_t i = 0; i < this->branch_codes.size(); i++) {
			if (this->branch_codes[i] != right_hand_side.branch_codes[i]) { return false; }
		}
		return true;
	}

	
	void Branch::ConvertIntoCanonicalRepresentation() {
		std::sort(branch_codes.begin(), branch_codes.end());
	}

	
	bool Branch::HasBranchedOnFeature(int feature) const {
		int code0 = GetCode(feature, true);
		int code1 = GetCode(feature, false);
		for (int i = 0; i < branch_codes.size(); i++) {
			if (branch_codes[i] == code0 || branch_codes[i] == code1) return true;
		}
		return false;
	}
}
