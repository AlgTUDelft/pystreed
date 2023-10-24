/**
Partly from Emir Demirovic "MurTree bi-objective"
https://bitbucket.org/EmirD/murtree-bi-objective
*/
#include "solver/counter.h"

namespace STreeD {
	Counter::Counter(int num_features) : num_features(num_features) {
		ResetToZeros();
	}

	int Counter::GetCount(int index_row, int index_column) const {
		runtime_assert(index_row <= index_column);
		int index = IndexSymmetricMatrix(index_row, index_column);
		return counts[index];
	}

	void Counter::ResetToZeros() {
		counts = std::vector<int>(NumElements(), 0);
		total_count = 0;
	}

	bool Counter::operator==(const Counter& reference)  const {
		if (num_features != reference.num_features) { return false; }
		if (total_count != reference.total_count) { return false; }
		for (int i = 0; i < NumElements(); i++) {
			if (counts[i] != reference.counts[i]) return false;
		}
		return true;
	}

	int Counter::NumElements() const {
		return num_features * (num_features + 1) / 2; //recall that the matrix is symmetric
	}

	int Counter::IndexSymmetricMatrix(int index_row, int index_column)  const {
		runtime_assert(index_row <= index_column);
		return num_features * index_row + index_column - index_row * (index_row + 1) / 2;
	}
}