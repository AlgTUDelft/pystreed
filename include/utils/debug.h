#pragma once
#include "base.h"

#ifndef DEBUG
#define DEBUG false
#endif

#define DEBUG_PRINT false

namespace STreeD {

	/* DEBUG functions */
	void PrintIndent(int depth) {
		if (!DEBUG_PRINT) return;
		for (int i = 0; i < depth; i++) std::cout << "  ";
	}

	void DebugBranch(const Branch& branch) {
		if (!DEBUG_PRINT) return;
		PrintIndent(branch.Depth());
		std::cout << branch << std::endl;
	}

	void DebugBranch(const Branch& branch, int feature) {
		if (!DEBUG_PRINT) return;
		PrintIndent(branch.Depth() + 1);
		std::cout << branch << " > " << feature << std::endl;
	}

	bool IsBranch(const Branch& branch, std::vector<int> features) {
		if (branch.Depth() != features.size()) return false;
		for (int i = 0; i < branch.Depth(); i++) {
			if (branch[i] != features[i]) return false;
		}
		return true;
	}

	void Pause(const Branch& branch, std::vector<int> features) {
		if (!DEBUG_PRINT) return;
		if (IsBranch(branch, features)) {
			std::cout << "Pause" << std::endl;
		}
	}

}