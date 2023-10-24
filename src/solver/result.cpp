#include "solver/result.h"
#include "tasks/tasks.h"

namespace STreeD {

	int SolverResult::GetBestDepth() const {
		return depths[best_index];
	}

	int SolverResult::GetBestNodeCount() const {
		return num_nodes[best_index];
	}

	/*Performance Performance::GetAverage(const std::vector<Performance>& performances) {
		runtime_assert(performances.size() > 0);
		const int n_objectives = performances[0].train_result.scores.size();
		const int n_test_objectives = performances[0].test_result.scores.size();
		const int n_classes = performances[0].train_result.class_count.size();
		Performance result(n_objectives, n_test_objectives, n_classes);
		for (int i = 0; i < n_objectives; i++) {
			for (const auto& p : performances) {
				result.train_result.scores[i] += p.train_result.scores[i];
			}
			result.train_result.scores[i] /= performances.size();
		}
		for (int i = 0; i < n_test_objectives; i++) {
			for (const auto& p : performances) {
				result.test_result.scores[i] += p.test_result.scores[i];
			}
			result.test_result.scores[i] /= performances.size();
		}

		result.train_result.metrics = std::vector<double>(performances[0].train_result.metrics.size(), 0);
		for (int i = 0; i < result.train_result.metrics.size(); i++) {
			result.train_result.metrics[i] = 0;
			for (const auto& p : performances) {
				result.train_result.metrics[i] += p.train_result.metrics[i];
			}
			result.train_result.metrics[i] /= performances.size();
		}
		result.test_result.metrics = std::vector<double>(performances[0].test_result.metrics.size(), 0);
		for (int i = 0; i < result.test_result.metrics.size(); i++) {
			result.test_result.metrics[i] = 0;
			for (const auto& p : performances) {
				result.test_result.metrics[i] += p.test_result.metrics[i];
			}
			result.test_result.metrics[i] /= performances.size();
		}
		for (const auto& p : performances) {
			result.train_result.average_path_length += p.train_result.average_path_length;
			result.test_result.average_path_length += p.test_result.average_path_length;
			for (int c = 0; c < n_classes; c++) {
				result.train_result.class_count[c] += p.train_result.class_count[c];
				result.test_result.class_count[c] += p.test_result.class_count[c];
			}
		}
		result.train_result.average_path_length /= performances.size();
		result.test_result.average_path_length /= performances.size();
		for (int c = 0; c < n_classes; c++) {
			result.train_result.class_count[c] /= performances.size();
			result.test_result.class_count[c] /= performances.size();
		}
		return result;
	}
	*/

}