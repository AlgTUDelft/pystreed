/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
Partly from Jacobus G.M. van der Linden "DPF"
https://gitlab.tudelft.nl/jgmvanderlinde/dpf
*/
#pragma once
#include "base.h"
#include "solver/optimization_utils.h"
#include "solver/feature_selector.h"
#include "solver/result.h"
#include "solver/tree.h"
#include "solver/cache.h"
#include "solver/terminal_solver.h"
#include "solver/difference_computer.h"
#include "solver/similarity_lowerbound.h"
#include "solver/statistics.h"
#include "tasks/tasks.h"
#include "utils/parameter_handler.h"
#include "utils/file_reader.h"
#include "utils/stopwatch.h"
#include "model/data.h"
#include "model/branch.h"
#include "model/node.h"
#include "model/container.h"


#define MAX_DEPTH 20

namespace STreeD {
	
	struct SolverParameters {
		SolverParameters(const ParameterHandler& parameters);
		bool verbose{ true };
		bool use_terminal_solver{ true };
		bool use_lower_bounding{ true };
		bool use_task_lower_bounding{ true };
		bool subtract_ub{ true };
		bool use_upper_bounding{ true };
		bool similarity_lb{ true };
		bool cache_data_splits{ false };
		bool use_lower_bound_early_stop{ true };
		int minimum_leaf_node_size{ 1 };
		size_t UB_LB_max_size{ 12 };
	};

	struct ProgressTracker {
		ProgressTracker(int num_features);

		void UpdateProgressCount(int count);
		void Done();
		void Reset() { count = 0; }

		int count = { 0 };
		int print_progress_every{ 1 };
		int progress_length{ 1 };
		int num_features{ 1 };
	};

	class AbstractSolver {
	public:
		AbstractSolver(ParameterHandler& parameters, std::default_random_engine* rng);
		void UpdateParameters(const ParameterHandler& parameters);
		inline const ParameterHandler& GetParameters() const { return parameters; }
		virtual std::shared_ptr<SolverResult> Solve(const ADataView& train_data) = 0;
		virtual std::shared_ptr<SolverResult> HyperSolve(const ADataView& train_data) = 0;
		virtual std::shared_ptr<SolverResult> TestPerformance(const std::shared_ptr<SolverResult>& result, const ADataView& test_data) = 0;
		virtual void InitializeTest(const ADataView& test_data, bool reset = false) = 0;
		inline void SetVerbosity(bool verbose) { solver_parameters.verbose = verbose; }
		virtual void PreprocessData(AData& data, bool train = true) = 0;

		inline const SolverParameters& GetSolverParameters() const { return solver_parameters; }
		inline int NumLabels() const { return train_data.NumLabels(); }
		inline int NumFeatures() const { return train_data.NumFeatures(); }
		inline bool IsTerminalNode(int depth, int num_nodes) { return solver_parameters.use_terminal_solver && depth <= 2; }
		inline std::string GetTaskString() const { return parameters.GetStringParameter("task"); }
		inline const std::vector<int>& GetFeatureOrder() const { return feature_order; }
		

	protected:
		SolverParameters solver_parameters;
		ParameterHandler parameters;

		ADataView org_train_data, train_data, org_test_data, test_data;
		DataSummary train_summary, test_summary;
		DataSplitter data_splitter;
		std::vector<int> feature_order;
		
		int max_depth_finished;
		bool reconstructing;

		Statistics stats;
		Stopwatch stopwatch;
		ProgressTracker progress_tracker;
		std::default_random_engine* rng;
	};

	template <class OT>
	using TerminalSolver_C = typename std::conditional < OT::use_terminal, TerminalSolver<OT>*, void*>::type;

	template <class OT>
	using SimilarityLowerBound_C = typename std::conditional < OT::element_additive, SimilarityLowerBoundComputer<OT>*, void*>::type;

	template <class OT>
	class Solver : public AbstractSolver {	
	public:

		using SolType = typename OT::SolType;			// The type of the solution value. E.g., int for misclassification score, double for sum of squared errors
		using SolContainer = typename std::conditional<OT::total_order, Node<OT>, std::shared_ptr<Container<OT>>>::type;
		using Context = typename OT::ContextType;		// The class type of the context (default = BranchContext)
		using LabelType = typename OT::LabelType;		// The class of the (input) label, e.g., double for regression, int for classification
		using SolLabelType = typename OT::SolLabelType; // The class of the leaf label, e.g., int for classification, or linear model for piecewise linear regression
		static constexpr bool sparse_objective = OT::total_order && OT::has_branching_costs
			&& OT::constant_branching_costs && (std::is_same<typename OT::SolType, double>::value || std::is_same<typename OT::SolType, int>::value);

		Solver(ParameterHandler& parameters, std::default_random_engine* rng);
		~Solver();

		/*
		* Initialize
		* 1) the optimization task
		* 2) the cache
		* 3) the data split cache
		* 4) the train data preprocessing
		* 5) the terminal solver
		* 6) the similarity lower bound computer
		* 7) the global upper bound
		*/
		void InitializeSolver(const ADataView& train_data, bool reset=false);

		/*
		* Initialize
		* 1) the test data preprocessing
		* 2) the data split cache
		*/
		void InitializeTest(const ADataView& test_data, bool reset = false);
		
		/*
		* Reset the cache
		*/
		void ResetCache();

		/*
		* Reset the cache (after going to a higher node/depth limit)
		*/
		void IncrementalResetCache();

		/*
		* Find the optimal tree for the given training data 
		*/
		std::shared_ptr<SolverResult> Solve(const ADataView& train_data);
		
		/*
		* Hypertune the parameters as specified by the optimization task
		* Find the optimal tree using hypertuning for the given training data and evaluate on the training and test data
		*/
		std::shared_ptr<SolverResult> HyperSolve(const ADataView& train_data);

		/*
		* Returns the test performance over test data given the trees in result
		*/
		std::shared_ptr<SolverResult> TestPerformance(const std::shared_ptr<SolverResult>& result, const ADataView& test_data);

		/*
		* Returns labels for test data given the tree
		*/
		std::vector<LabelType> Predict(const std::shared_ptr<Tree<OT>>& tree, const ADataView& test_data);

		/*
		* Solve an independent subtree for the given data and context with at most size limits given by max_depth and num_nodes
		* Only return solutions that are not dominated by the upper bound UB
		*/
		SolContainer SolveSubTree(ADataView& data, const Context& context, SolContainer UB, int max_depth, int num_nodes);

		/*
		* Apply a recursive step in the search for optimal trees: split on all possible features and retain all non-dominated solutions
		*/
		SolContainer SolveSubTreeGeneralCase(ADataView& data, const Context& context, SolContainer& UB, SolContainer& solutions, int max_depth, int num_nodes);

		/*
		* Solve a subtree using the special terminal solver 
		* (if supported by the optimization task)
		*/
		template <typename U = OT, typename std::enable_if<U::use_terminal, int>::type = 0>
		SolContainer SolveTerminalNode(ADataView& data, const Context& context, SolContainer& UB, int max_depth, int num_nodes);

		/*
		* Solve a leaf node
		*/
		SolContainer SolveLeafNode(const ADataView& data, const Context& context, SolContainer& UB) const;

		/*
		* Merge Pareto-fronts from a left and right subtree
		* in a given context
		* when splitting on feature, with branching costs
		* store the result in final_sols
		* if reconstruct=true, store the result in tree instead.
		*/
		template <bool reconstruct=false, typename U = OT, typename std::enable_if<!U::total_order, int>::type = 0>
		void Merge(int feature, const Context& context, SolContainer& UB, SolContainer& left_sols, SolContainer& right_sols, const SolType& branching_costs, SolContainer& final_sols,
			TreeNode<U>* tree = nullptr);

		/*
		* Merge two Pareto-front lower bounds
		* Limit the size of the new lower bound to improve computation speed
		*/
		template <typename U = OT, typename std::enable_if<!U::total_order, int>::type = 0>
		void LBMerge(int feature, const Context& context, SolContainer& left_sols, SolContainer& right_sols, const SolType& branching_costs, SolContainer& final_sols);

		/*
		* Compute a lower bound
		*/
		void ComputeLowerBound(ADataView& data, const Context& context, SolContainer& lb_out, int depth, int num_nodes, bool root_is_branching_node);

		/*
		* Compute a lower bound from the combination of the left and right lower bound
		*/
		void ComputeLeftRightLowerBound(int feature, const Context& context, const SolType& branching_costs,
			SolContainer& lb_out, SolContainer& lb_left_out, SolContainer& lb_right_out,
			ADataView& left_data, const Context& left_context, int left_depth, int left_nodes,
			ADataView& right_data, const Context& right_context, int right_depth, int right_nodes);

		/*
		* Given an upper bound UB, substract the solutions sols and the branching costs from it using the substract operator 
		* and store the result in updatedUB
		* If the current node is the root node, and the task has constraints, relax the solutions for better UBs
		*/
		void SubtractUBs(const Context& context, const SolContainer& UB, const SolContainer& sols, const SolContainer& current_solutions, const SolType& branching_costs, SolContainer& updatedUB);

		/*
		* Returns true iff the solution is feasible
		*/
		bool SatisfiesConstraint(const Node<OT>& sol, const Context& context) const;
		
		/*
		* Update the upper bound
		*/
		void UpdateUB(const Context& context, SolContainer& UB, Node<OT> sol) const;

		/*
		* Get the weight of a data set
		*/
		int GetDataWeight(const ADataView& data) const;

		/*
		* Check the minimum leaf node size
		*/
		bool SatisfiesMinimumLeafNodeSize(const ADataView& data, int multiplier = 1) const;

		/*
		* Construct the tree for a given solution from the cache
		*/
		std::shared_ptr<Tree<OT>> ConstructOptimalTree(const Node<OT>& sol, ADataView& data, const Context& context, int max_depth, int num_nodes);
		
		/*
		* Search for other cached solutions that are based on datasets similar to this dataset.
		* Update lower bounds in the cache using the similarity lower bound if appropriate.
		* returns true iff the method updated the optimal solution of the branch; 
		*/
		template <typename U = OT, typename std::enable_if<U::element_additive, int>::type = 0>
		bool UpdateCacheUsingSimilarity(ADataView& data, const Branch& branch, int max_depth, int num_nodes);

		/*
		* Get the branching costs for branching on feature in context
		*/
		SolType GetBranchingCosts(const ADataView& data, const Context& context, int feature) const;

		/*
		* Get the task
		*/
		inline OT* GetTask() const { return task; }

		/*
		* Preprocess the data
		*/
		void PreprocessData(AData& data, bool train = true);

		/*
		* Preprocess the training data
		*/
		void PreprocessTrainData(const ADataView& org_train_data, ADataView& train_data);
		
		/*
		* Preprocess the training data
		*/
		void PreprocessTestData(const ADataView& org_test_data, ADataView& test_data);

		/*
		* Postprocess the tree
		*/
		void PostProcessTree(std::shared_ptr<Tree<OT>> tree);

		/*
		* Reduce the number of nodes and depth limit based on a constant branching costs
		*/
		void ReduceNodeBudget(const ADataView& data, const Context& context, const SolContainer& UB, int& max_depth, int& num_nodes) const;

		/*
		* Return true if a feature is redundant
		*/
		bool IsRedundantFeature(int feature) const { return redundant_features[feature]; }

	private:
		OT* task{ nullptr };
		Cache<OT>* cache{ nullptr };
		
		TerminalSolver_C<OT> terminal_solver1{ nullptr }, terminal_solver2{ nullptr };
		SimilarityLowerBound_C<OT> similarity_lower_bound_computer{ nullptr };
		SolContainer global_UB;
		std::vector<int> flipped_features;
		std::vector<int> redundant_features;
	};
}