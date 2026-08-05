/**
Partly from Emir Demirovic "MurTree bi-objective"
https://bitbucket.org/EmirD/murtree-bi-objective
*/
#pragma once
#include "base.h"

namespace STreeD {

	// Store the counts (number of instances) of all four leaf nodes of a depth-two tree
	struct Counts {
		int count00{ 0 };
		int count01{ 0 };
		int count10{ 0 };
		int count11{ 0 };
	};

	// Counter counts the number of instances in a dataset for each possible combination of features
	class Counter {
	public:
		Counter() : num_features(0), counts(0), total_count(0) {}
		Counter(int num_features);

		// Get the count for feature 'index_row' and feature 'index_column'
		int GetCount(int index_row, int index_column) const;
		
		// Get the count for index. This function assumes index is computed by using IndexSymmetricMatrix
		inline int GetCount(int index) const { return counts[index]; }
		
		// Update (add) the count for feature 'index_row' and feature 'index_column'
		inline void UpdateCount(int index_row, int index_column, int val) { counts[IndexSymmetricMatrix(index_row, index_column)] += val;  runtime_assert(counts[IndexSymmetricMatrix(index_row, index_column)] >= 0);
		}
		
		// Update (add) the count for feature 'index_row_col' (row=col)
		inline void UpdateCount(int index_row_col, int val) { counts[IndexSymmetricMatrix(index_row_col)] += val;  runtime_assert(counts[IndexSymmetricMatrix(index_row_col)] >= 0);
		}
		
		// Update (add) the count for feature 'index'. 
		// This function assumes index is computed by using IndexSymmetricMatrix
		inline void UpdateCountByIndex(int index, int val) { counts[index] += val; runtime_assert(counts[index] >= 0); }
		
		// Update the total count (before splitting)
		inline void UpdateTotalCount(int count) { total_count += count; runtime_assert(total_count >= 0); }
		
		// Reset all counts to zero
		void ResetToZeros();
		
		// Return the total count (before splitting)
		int GetTotalCount() const { runtime_assert(total_count >= 0); return total_count; }

		bool operator==(const Counter& reference)  const;

	private:
		
		// Return the size of the count vector
		int NumElements() const;
		
		// Compute the 1D index from the 2D index specification
		int IndexSymmetricMatrix(int index_row, int index_column)  const;
		
		// Compute the 1D index from the 2D index specification (col=row)
		int IndexSymmetricMatrix(int index_row_column)  const;

		//Store the counts in 1D. 
		std::vector<int> counts;
		int num_features;
		int total_count;
	};
}