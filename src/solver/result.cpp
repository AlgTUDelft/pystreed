#include "solver/result.h"
#include "tasks/tasks.h"

namespace STreeD {

	int SolverResult::GetBestDepth() const {
		return depths[best_index];
	}

	int SolverResult::GetBestNodeCount() const {
		return num_nodes[best_index];
	}

}