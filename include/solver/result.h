#pragma once
#include "model/container.h"
#include "solver/tree.h"
#include "solver/data_splitter.h"

namespace STreeD {
	
	class OptimizationTask;

	struct Score {
		using ScoreType = double;
		
		Score() = default;

		ScoreType score{ 0 };			// the test score obtained on the data
		double average_path_length{ 0 };// the average path (question) length
		
		/*
		Compute the average scores over multiple performances
		*/
		static std::shared_ptr<Score> GetAverage(const std::vector<std::shared_ptr<Score>>& performances) {
			runtime_assert(performances.size() > 0);
			auto avg = std::make_shared<Score>();
			for (auto& p : performances) {
				avg->score += p->score;
				avg->average_path_length += p->average_path_length;
			}
			avg->score /= performances.size();
			avg->average_path_length /= performances.size();
			return avg;
		}

	};

	template <class OT>
	struct InternalTrainScore : public Score {
		using SolType = typename OT::SolType;
		using TestSolType = typename OT::TestSolType;

		InternalTrainScore() = default;

		static std::shared_ptr<Score> ComputeTrainPerformance(DataSplitter* data_splitter, OT* task, const Tree<OT>* tree, const ADataView& train_data) {
			auto result = std::make_shared<InternalTrainScore>();
			typename OT::ContextType context;
			
			// recursively compute the score on the given data
			tree->ComputeTrainScore(data_splitter, task, context, train_data, *result);
			
			// compute the final train and test score
			result->score = task->ComputeTrainTestScore(result->train_test_value);
			result->average_path_length = result->average_path_length / double(train_data.Size());
			return result;
		}

		static std::shared_ptr<Score> GetWorst(OT* task) {
			auto result = std::make_shared<InternalTrainScore>();
			result->train_value = OT::worst;
			result->train_test_value = OT::worst; // replace with test worst
			result->score = task->ComputeTrainTestScore(result->train_test_value);
			return result;
		}

		// Solution value for each solution tree on the training data, e.g., misclassification score
		SolType train_value;
		// Test solution value for each solution tree on the training data, e.g., f1-score
		TestSolType train_test_value;
	};

	template <class OT>
	struct InternalTestScore : public Score {
		using TestSolType = typename OT::TestSolType;

		InternalTestScore() = default;

		static std::shared_ptr<Score> ComputeTestPerformance(DataSplitter* data_splitter, OT* task, const Tree<OT>* tree,
				const std::vector<int>& flipped_features, const ADataView& test_data) {
			auto result = std::make_shared<InternalTestScore>();
			typename OT::ContextType context;

			// recursively compute the score on the given data
			tree->ComputeTestScore(data_splitter, task, context, flipped_features, test_data, *result);

			// compute the final train and test score
			result->score = task->ComputeTestTestScore(result->test_test_value);
			result->average_path_length = result->average_path_length / double(test_data.Size());
			return result;
		}

		static std::shared_ptr<Score> GetWorst(OT* task) {
			auto result = std::make_shared<InternalTestScore>();
			result->test_test_value = OT::worst;  // replace with test worst
			result->score = task->ComputeTestTestScore(result->test_test_value);
			return result;
		}

		// Test solution value for each solution tree on the test data, e.g., f1-score
		TestSolType test_test_value;
	};

	struct SolverResult : public std::enable_shared_from_this<SolverResult> {

		SolverResult() = default;

		inline bool IsFeasible() const { return scores.size() > 0; }
		inline bool IsProvenOptimal() const { return is_proven_optimal; }
		int GetBestDepth() const;
		int GetBestNodeCount() const;
		inline size_t NumSolutions() const { return scores.size(); }
		inline void SetScore(size_t index, const std::shared_ptr<Score>& score) { scores[index] = score; }

		bool is_proven_optimal{ false };
		std::vector<std::shared_ptr<Score>> scores;
		size_t best_index{ 0 };
		std::vector<int> depths, num_nodes;
		std::vector<std::string> tree_strings;
	};

	template <class OT>
	struct SolverTaskResult : public SolverResult {
		SolverTaskResult() = default;

		void AddSolution(std::shared_ptr<Tree<OT>> tree, std::shared_ptr<Score> score) {
			size_t i;
			for (i = 0; i < scores.size(); i++) {
				if (scores[i]->score > score->score) break;
			}
			trees.insert(trees.begin() + i, tree);
			scores.insert(scores.begin() + i, score);
			depths.insert(depths.begin() + i, tree->Depth());
			num_nodes.insert(num_nodes.begin() + i, tree->NumNodes());
			tree_strings.insert(tree_strings.begin() + i, tree->ToString());
		}

		std::vector<std::shared_ptr<Tree<OT>>> trees;
	};

}