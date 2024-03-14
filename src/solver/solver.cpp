#include "solver/solver.h"
#include "utils/debug.h"

namespace STreeD {

	/************************
	*    SolverParameters   *
	************************/

	SolverParameters::SolverParameters(const ParameterHandler& parameters) :
		verbose(parameters.GetBooleanParameter("verbose")),
		use_terminal_solver(parameters.GetBooleanParameter("use-terminal-solver")),
		use_upper_bounding(parameters.GetBooleanParameter("use-upper-bound")),
		use_lower_bounding(parameters.GetBooleanParameter("use-lower-bound")),
		similarity_lb(parameters.GetBooleanParameter("use-similarity-lower-bound")),
		minimum_leaf_node_size(int(parameters.GetIntegerParameter("min-leaf-node-size"))) { }

	/************************
	*    AbstractSolver     *
	************************/

	AbstractSolver::AbstractSolver(ParameterHandler& parameters, std::default_random_engine* rng) :
		parameters(parameters), solver_parameters(parameters), rng(rng),
		data_splitter(MAX_DEPTH) { }

	void AbstractSolver::UpdateParameters(const ParameterHandler& parameters) {
		this->parameters = parameters;
		solver_parameters = SolverParameters(parameters);
	}

	/************************
	*        Solver         *
	************************/

	template<class OT>
	Solver<OT>::Solver(ParameterHandler& parameters, std::default_random_engine* rng) : AbstractSolver(parameters, rng) {
		task = new OT(parameters);
	}

	template<class OT>
	Solver<OT>::~Solver() {
		if (cache != nullptr) delete cache;
		if constexpr (OT::use_terminal) {
			if (terminal_solver1 != nullptr) delete terminal_solver1;
			if (terminal_solver2 != nullptr) delete terminal_solver2;
		}
		if constexpr (OT::element_additive) {
			if (similarity_lower_bound_computer != nullptr) delete similarity_lower_bound_computer;
		}
		if (task != nullptr) delete task;
	}

	template<class OT>
	std::shared_ptr<SolverResult> Solver<OT>::Solve(const ADataView& _train_data) {
		stopwatch.Initialise(parameters.GetFloatParameter("time"));
		InitializeSolver(_train_data);

		Solver<OT>::Context root_context;

		// if no UB is given yet, compute the leaf solutions in the root node and use these as UB
		auto result = InitializeSol<OT>();
		if (CheckEmptySol<OT>(global_UB)) {
			global_UB = InitializeSol<OT>();
			// If an upper bound is provided, and the objective is numeric, add it to the UB
			if constexpr (std::is_same<Solver<OT>::SolType, double>::value || std::is_same<Solver<OT>::SolType, int>::value) {
				AddSol<OT>(global_UB, Node<OT>(parameters.GetFloatParameter("upper-bound")));
			}
			result = SolveLeafNode(train_data, root_context, global_UB);
		}

		// If all number of nodes options should be considered, set min_num_nodes to 1, else to max
		int min_num_nodes = int(parameters.GetIntegerParameter("max-num-nodes"));
		if (parameters.GetBooleanParameter("all-trees")) { min_num_nodes = 1; }

		// For each number of nodes that should be considered, find the optimal solution
		for (int num_nodes = min_num_nodes; num_nodes <= int(parameters.GetIntegerParameter("max-num-nodes")); num_nodes++) {
			if (!stopwatch.IsWithinTimeLimit()) { break; }
			int max_depth = std::min(int(parameters.GetIntegerParameter("max-depth")), num_nodes);
			auto sol = SolveSubTree(train_data, root_context, global_UB, max_depth, num_nodes );
			AddSols<OT>(task, 0, result, sol);
		}
		
		// Evaluate the results
		auto solver_result = std::make_shared<SolverTaskResult<OT>>();
		solver_result->is_proven_optimal = stopwatch.IsWithinTimeLimit();
		if constexpr (OT::total_order) {
			if (result.IsFeasible()) {
				clock_t clock_start = clock();
				auto tree = ConstructOptimalTree(result, train_data, root_context, int(parameters.GetIntegerParameter("max-depth")), result.NumNodes());
				stats.time_reconstructing += double(clock() - clock_start) / CLOCKS_PER_SEC;
				auto score = InternalTrainScore<OT>::ComputeTrainPerformance(&data_splitter, task, tree.get(), train_data);
				tree->FlipFlippedFeatures(flipped_features);
				solver_result->AddSolution(tree, score);
			}
		} else {
			for (auto& s : result->GetSolutions()) {
				clock_t clock_start = clock();
				auto tree = ConstructOptimalTree(s, train_data, root_context, int(parameters.GetIntegerParameter("max-depth")), int(parameters.GetIntegerParameter("max-num-nodes")));
				stats.time_reconstructing += double(clock() - clock_start) / CLOCKS_PER_SEC;
				auto score = InternalTrainScore<OT>::ComputeTrainPerformance(&data_splitter, task, tree.get(), train_data);
				tree->FlipFlippedFeatures(flipped_features);
				solver_result->AddSolution(tree, score);
			}
		}

		stats.total_time += stopwatch.TimeElapsedInSeconds();
		if (solver_parameters.verbose) {
			stats.Print();
		}

		return solver_result;
	}

	template<class OT>
	void Solver<OT>::InitializeSolver(const ADataView& _train_data, bool reset) {
		// Inform the task about the solver parameters
		task->UpdateParameters(parameters);

		// If the training data is the same, the cache does not need to be repopulated
		// Except if the hyper tune configuration sets reset to true
		if (!reset && org_train_data == _train_data) return;
		
		// Update the data objects, and inform the task about the new data
		org_train_data = _train_data;
		PreprocessTrainData(org_train_data, train_data);
		train_summary = DataSummary(train_data);
		task->InformTrainData(train_data, train_summary);

		// Reset the cache, data split cache, terminal solvers, sim-bound computer
		if (cache != nullptr) delete cache;
		if constexpr (OT::use_terminal) {
			if (terminal_solver1 != nullptr) delete terminal_solver1;
			if (terminal_solver2 != nullptr) delete terminal_solver2;
		}
		if constexpr (OT::element_additive) {
			if (similarity_lower_bound_computer != nullptr) delete similarity_lower_bound_computer;
		}
		cache = new Cache<OT>(parameters, MAX_DEPTH, train_data.Size());
		if (!solver_parameters.use_lower_bounding) cache->DisableLowerBoundCaching();
		if constexpr (OT::use_terminal) {
			terminal_solver1 = new TerminalSolver<OT>(this);
			terminal_solver2 = new TerminalSolver<OT>(this);
		}
		// Only use the similarity lower bound computer if the optimization task is element-wise additive
		if constexpr (OT::element_additive) {
			similarity_lower_bound_computer = new SimilarityLowerBoundComputer<OT>(task,
				NumLabels(), MAX_DEPTH, int(parameters.GetIntegerParameter("max-num-nodes")), train_data.Size());
			if (!solver_parameters.similarity_lb) similarity_lower_bound_computer->Disable();
		} else {
			solver_parameters.similarity_lb = false;
		}

		// Disable data split cache, if so configured
		if (!solver_parameters.cache_data_splits) data_splitter.Disable();
		data_splitter.Clear();

		// Initialize the global UB with an empty solution
		global_UB = InitializeSol<OT>();
	}
	
	template<class OT>
	void Solver<OT>::InitializeTest(const ADataView& _test_data, bool reset) {
		// If the training data is the same, the cache does not need to be repopulated
		// Except if the hyper tune configuration sets reset to true
		if (!reset && org_test_data == _test_data) return;

		// Update the data objects, and inform the task about the new data
		org_test_data = _test_data;
		PreprocessTestData(org_test_data, test_data);
		test_summary = DataSummary(test_data);
		task->InformTestData(test_data, test_summary);

		data_splitter.Clear(true);
	}

	template <class OT>
	typename Solver<OT>::SolContainer Solver<OT>::SolveSubTree(ADataView & data, const Solver<OT>::Context& context, typename Solver<OT>::SolContainer UB_, int max_depth, int num_nodes) {
		runtime_assert(0 <= max_depth && max_depth <= num_nodes);
		if (!stopwatch.IsWithinTimeLimit()) { return InitializeSol<OT>(); }

		const Branch& branch = context.GetBranch();
		auto UB = CopySol<OT>(UB_); // Make a copy  of the UB, because we are going to update it, but these updates should not be passed to higher nodes

		ReduceNodeBudget(data, context, UB, max_depth, num_nodes);

		if (max_depth == 0 || num_nodes == 0) {
			return SolveLeafNode(data, context, UB);
		}

		// Check Cache
		{
			auto results = cache->RetrieveOptimalAssignment(data, branch, max_depth, num_nodes);
			if (!CheckEmptySol<OT>(results)) {
				return results;
			}
		}

		auto leaf_solutions = InitializeSol<OT>();
		if (solver_parameters.use_lower_bounding) {
			if constexpr (OT::element_additive) {
				// Update the cache using the similarity-based lower bound
				// If an optimal solution was found in the process, return it
				bool updated_optimal_solution = UpdateCacheUsingSimilarity(data, branch, max_depth, num_nodes);
				if (updated_optimal_solution) {
					auto results = cache->RetrieveOptimalAssignment(data, branch, max_depth, num_nodes);
					if (!CheckEmptySol<OT>(results)) {
						//stats.num_cache_hit_optimality++;
						return results;
					}
				}
			}

			//Check LB > UB and return infeasible if true
			auto lower_bound = cache->RetrieveLowerBound(data, branch, max_depth, num_nodes);
			
			if (solver_parameters.use_upper_bounding && LeftStrictDominatesRight<OT>(UB, lower_bound)) {
				return InitializeSol<OT>();
			}

			//Check lower-bound vs leaf node solution and return if same
			auto empty_UB = InitializeSol<OT>();
			leaf_solutions = SolveLeafNode(data, context, empty_UB);
			if (SolutionsEqual<OT>(lower_bound, leaf_solutions)) return leaf_solutions; 
		}

		// Use the specialised algorithm for small trees
		if constexpr (OT::use_terminal) {
			if (IsTerminalNode(max_depth, num_nodes)) {
				if (solver_parameters.use_upper_bounding) AddRootRelaxSols<OT>(task, branch, UB, leaf_solutions);
				return SolveTerminalNode(data, context, UB, max_depth, num_nodes);
			}
		}

		// In all other cases, run the recursive general case
		return SolveSubTreeGeneralCase(data, context, UB_, max_depth, num_nodes);
	}

	template <class OT>
	typename Solver<OT>::SolContainer Solver<OT>::SolveSubTreeGeneralCase(ADataView& data, const Solver<OT>::Context& context, typename Solver<OT>::SolContainer& UB, int max_depth, int num_nodes) {
		runtime_assert(max_depth <= num_nodes);
		
		const Branch& branch = context.GetBranch();
		auto orgUB = CopySol<OT>(UB); // Copy the original UB and keep it, in case this branch is infeasible. The original UB can then be stored as LB
		auto infeasible_lb = InitializeSol<OT>();
		int current_depth = branch.Depth();
		const int max_size_subtree = std::min((1 << (max_depth - 1)) - 1, num_nodes - 1); //take the minimum between a full tree of max_depth or the number of nodes - 1
		const int min_size_subtree = num_nodes - 1 - max_size_subtree;
		typename Solver<OT>::SolContainer lb, left_lower_bound, right_lower_bound;

		auto solutions = SolveLeafNode(data, context, UB);
		
		if (data.Size() < 2 * solver_parameters.minimum_leaf_node_size)
			return solutions;

		auto branch_lb = solver_parameters.use_lower_bounding
			? cache->RetrieveLowerBound(data, branch, max_depth, num_nodes)
			: InitializeSol<OT>(true);

		// Initialize the feature selector
		std::unique_ptr<FeatureSelectorAbstract> feature_selector;
		if (parameters.GetStringParameter("feature-ordering") == "in-order") {
			feature_selector = std::make_unique<FeatureSelectorInOrder>(data.NumFeatures());
		} else if (parameters.GetStringParameter("feature-ordering") == "gini") {
			runtime_assert(!(std::is_same<typename OT::LabelType, double>::value)); // Regression does not work with Gini
			feature_selector = std::make_unique<FeatureSelectorGini>(data.NumFeatures());
		}  else { std::cout << "Unknown feature ordering strategy!" << std::endl; exit(1); }
		feature_selector->Initialize(data);

		// Loop over each feature
		while (feature_selector->AreThereAnyFeaturesLeft()) {
			if (!stopwatch.IsWithinTimeLimit()) break;
			// if the current set of solutions equals the LB for this branch: break
			if (solver_parameters.use_lower_bounding && SolutionsEqual<OT>(branch_lb, solutions)) break;			
			if (solver_parameters.use_lower_bounding
				&& solver_parameters.use_upper_bounding
				&& LeftStrictDominatesRight<OT>(UB, branch_lb)) break;
			int feature = feature_selector->PopNextFeature();
			if (branch.HasBranchedOnFeature(feature) || !task->MayBranchOnFeature(feature)) continue;
			auto branching_costs = GetBranchingCosts(data, context, feature);
			// Break if the current UB is lower than the constant branching costs (if applicable)
			if constexpr (OT::total_order && OT::has_branching_costs && OT::constant_branching_costs
				&& std::is_same<typename OT::SolType, double>::value || std::is_same<typename OT::SolType, int>::value) {
				if (solver_parameters.use_upper_bounding && UB.solution < branching_costs) break;				
			}

			// Split the data and skip if the split does not meet the minimum leaf node size requirements
			ADataView left_data;
			ADataView right_data;
			data_splitter.Split(data, branch, feature, left_data, right_data);
			if (left_data.Size() < solver_parameters.minimum_leaf_node_size || right_data.Size() < solver_parameters.minimum_leaf_node_size) continue;

			// Generate the context descriptors for the left and richt sub-branch
			Solver<OT>::Context left_context, right_context;
			task->GetLeftContext(data, context, feature, left_context);
			task->GetRightContext(data, context, feature, right_context);

			// switch the left and right branch if the left has more data
			ADataView* left_data_ptr = &left_data;
			ADataView* right_data_ptr = &right_data;
			Solver<OT>::Context* left_context_ptr = &left_context;
			Solver<OT>::Context* right_context_ptr = &right_context;
			bool swap_left_right = false;
			if (left_data.Size() < right_data.Size()) {
				swap_left_right = true;
				std::swap(left_data_ptr, right_data_ptr);
				std::swap(left_context_ptr, right_context_ptr);
			}
			const Branch& left_branch = left_context_ptr->GetBranch();
			const Branch& right_branch = right_context_ptr->GetBranch();

			// Loop over every possible way of distributing the node budget over the left and right subtrees
			for (int left_subtree_size = min_size_subtree; left_subtree_size <= max_size_subtree; left_subtree_size++) {
				int right_subtree_size = num_nodes - left_subtree_size - 1; //the '-1' is necessary since using the parent node counts as a node
				int left_depth = std::min(max_depth - 1, left_subtree_size);
				int right_depth = std::min(max_depth - 1, right_subtree_size);

				// Compute the left and right and combined LBs
				ComputeLeftRightLowerBound(feature, context, branching_costs, lb, left_lower_bound, right_lower_bound,
					*left_data_ptr, left_branch, left_depth, left_subtree_size, *right_data_ptr, right_branch, right_depth, right_subtree_size);
				if (solver_parameters.use_upper_bounding && LeftStrictDominatesRight<OT>(UB, lb)) {
					AddSols<OT>(infeasible_lb, lb);
					continue;
				}
				if (SolutionsEqual<OT>(lb, solutions)) continue;
				

				// substract the right LB from the UB to get a UB for the left branch
				auto leftUB = InitializeSol<OT>();
				SubtractUBs(context, UB, right_lower_bound, solutions, branching_costs, leftUB);

				// Solve the left branch
				auto left_solutions = SolveSubTree(*left_data_ptr, *left_context_ptr, leftUB, left_depth, left_subtree_size);

				if (!stopwatch.IsWithinTimeLimit()) break;
				if (CheckEmptySol<OT>(left_solutions)) {
					ComputeLeftRightLowerBound(feature, context, branching_costs, lb, left_lower_bound, right_lower_bound,
						*left_data_ptr, left_branch, left_depth, left_subtree_size, *right_data_ptr, right_branch, right_depth, right_subtree_size);
					AddSols<OT>(infeasible_lb, lb);
					continue;
				}

				// substract the left solutions from the UB to get a UB for the right branch
				auto rightUB = InitializeSol<OT>();
				SubtractUBs(context, UB, left_solutions, solutions, branching_costs, rightUB);

				// Solve the right branch
				auto right_solutions = SolveSubTree(*right_data_ptr, *right_context_ptr, rightUB, right_depth, right_subtree_size);

				if (!stopwatch.IsWithinTimeLimit()) break;
				if (CheckEmptySol<OT>(right_solutions)) {
					ComputeLeftRightLowerBound(feature, context, branching_costs, lb, left_lower_bound, right_lower_bound,
						*left_data_ptr, left_branch, left_depth, left_subtree_size, *right_data_ptr, right_branch, right_depth, right_subtree_size);
					AddSols<OT>(infeasible_lb, lb);
					continue;
				}

				// Combine left and right solutions and store the solution
				if constexpr (!OT::total_order) {
					if (swap_left_right) {
						Merge(feature, context, UB, right_solutions, left_solutions, branching_costs, solutions);
					} else {
						Merge(feature, context, UB, left_solutions, right_solutions, branching_costs, solutions);
					}
				} else {
					Node<OT> new_node;
					if (swap_left_right) {
						CombineSols(feature, right_solutions, left_solutions, branching_costs, new_node);
					} else {
						CombineSols(feature, left_solutions, right_solutions, branching_costs, new_node);
					}
					if (solver_parameters.use_upper_bounding && LeftStrictDominatesRightSol<OT>(UB, new_node)) continue;
					if (LeftStrictDominatesRightSol<OT>(new_node, solutions)) solutions = new_node;
					UpdateUB(context, UB, new_node);
				}

			}

		}

		// If a feasible solution is found (better than the UB), store it in the cache
		// Or update the LB for this branch
		SetSolSizeBudget<OT>(solutions, max_depth, num_nodes);
		if (!CheckEmptySol<OT>(solutions)) {
			cache->StoreOptimalBranchAssignment(data, branch, solutions, max_depth, num_nodes);
		} else {
			if (SolutionsEqual<OT>(infeasible_lb, InitializeSol<OT>())) {
				infeasible_lb = orgUB;
			} else {
				AddSolsInv<OT>(infeasible_lb, orgUB);
			}
			cache->UpdateLowerBound(data, branch, infeasible_lb, max_depth, num_nodes);
		}
		if constexpr (OT::element_additive) {
			similarity_lower_bound_computer->UpdateArchive(data, branch, max_depth);
		}

		return solutions;
	}

	template <class OT>
	template <typename U, typename std::enable_if<U::use_terminal, int>::type>
	typename Solver<OT>::SolContainer Solver<OT>::SolveTerminalNode(ADataView& data, const Solver<OT>::Context& context, typename Solver<OT>::SolContainer& UB, int max_depth, int num_nodes) {
		const Branch& branch = context.GetBranch();
		runtime_assert(max_depth <= 2 && 1 <= num_nodes && num_nodes <= 3 && max_depth <= num_nodes);
		runtime_assert(num_nodes != 3 || !cache->IsOptimalAssignmentCached(data, branch, 2, 3));
		runtime_assert(num_nodes != 2 || !cache->IsOptimalAssignmentCached(data, branch, 2, 2));
		runtime_assert(num_nodes != 1 || !cache->IsOptimalAssignmentCached(data, branch, 1, 1));

		stats.num_terminal_nodes_with_node_budget_one += (num_nodes == 1);
		stats.num_terminal_nodes_with_node_budget_two += (num_nodes == 2);
		stats.num_terminal_nodes_with_node_budget_three += (num_nodes == 3);

		DebugBranch(branch);

		// To maximize efficiency, use the terminal solver that already has computed frequency counts
		// for a dataset that is most similar to the new dataset
		clock_t clock_start = clock();
		int diff1 = terminal_solver1->ProbeDifference(data);
		int diff2 = terminal_solver2->ProbeDifference(data);
		TerminalSolver<OT>* tsolver = diff1 < diff2 ? terminal_solver1 : terminal_solver2;
		TerminalResults<OT>& results = tsolver->Solve(data, context, UB, num_nodes);
		stats.time_in_terminal_node += double(clock() - clock_start) / CLOCKS_PER_SEC;

		// Store solutions in the cache for different node budgets
		if (!cache->IsOptimalAssignmentCached(data, branch, 1, 1)) {
			auto& one_node_solutions = results.one_node_solutions;
			if (!CheckEmptySol<OT>(one_node_solutions)) {
				cache->StoreOptimalBranchAssignment(data, branch, one_node_solutions, 1, 1);
			} else {
				cache->UpdateLowerBound(data, branch, UB, 1, 1);
			}
		}
		if (!cache->IsOptimalAssignmentCached(data, branch, 2, 2)) {
			auto& two_nodes_solutions = results.two_nodes_solutions;
			if (!CheckEmptySol<OT>(two_nodes_solutions)) {
				cache->StoreOptimalBranchAssignment(data, branch, two_nodes_solutions, 2, 2);
			} else {
				cache->UpdateLowerBound(data, branch, UB, 2, 2);
			}
		}
		if (!cache->IsOptimalAssignmentCached(data, branch, 2, 3)) {
			auto& three_nodes_solutions = results.three_nodes_solutions;
			if (!CheckEmptySol<OT>(three_nodes_solutions)) {
				cache->StoreOptimalBranchAssignment(data, branch, three_nodes_solutions, 2, 3);
			} else {
				cache->UpdateLowerBound(data, branch, UB, 2, 3);
			}
		}
		if constexpr (OT::element_additive) {
			similarity_lower_bound_computer->UpdateArchive(data, branch, max_depth);
		}

		// Return solutions based on node budget
		if (num_nodes == 1) {
			if (LeftStrictDominatesRight<OT>(UB, results.one_node_solutions)) return InitializeSol<OT>();
			return CopySol<OT>(results.one_node_solutions);
		} else if (num_nodes == 2) {
			if (LeftStrictDominatesRight<OT>(UB, results.two_nodes_solutions)) return InitializeSol<OT>();
			return CopySol<OT>(results.two_nodes_solutions);
		}
		if (LeftStrictDominatesRight<OT>(UB, results.three_nodes_solutions)) return InitializeSol<OT>();
		return CopySol<OT>(results.three_nodes_solutions);
	}

	template<class OT>
	typename Solver<OT>::SolContainer Solver<OT>::SolveLeafNode(const ADataView& data, const Solver<OT>::Context& context, typename Solver<OT>::SolContainer& UB) const {
		if (data.Size() < solver_parameters.minimum_leaf_node_size) return InitializeSol<OT>();
		const Branch& branch = context.GetBranch();

		// If the optimization task has a custom leaf node function defined, use it
		if constexpr (OT::custom_leaf) {
			// Return the optimal feasible solution, requires custom implementation
			auto sol = task->SolveLeafNode(data, context);
			if (solver_parameters.use_upper_bounding && LeftStrictDominatesRightSol<OT>(UB, sol)) return InitializeSol<OT>();
			UpdateUB(context, UB, sol);
			return sol;
		} else { // Otherwise, for each possible label, calculate the costs and store if better
			auto result = InitializeSol<OT>();

			for (int label = 0; label < data.NumLabels(); label++) {
				auto sol = Node<OT>(label, task->GetLeafCosts(data, context, label));
				if (!SatisfiesConstraint(sol, context)) continue;
				if (solver_parameters.use_upper_bounding && LeftStrictDominatesRightSol<OT>(UB, sol)) continue;
				AddSol<OT>(task, branch.Depth(), result, sol);
				UpdateUB(context, UB, sol);
			}
			return result;
		}
	}

	template <class OT>
	void Solver<OT>::ComputeLeftRightLowerBound(int feature, const typename Solver<OT>::Context& context, const typename Solver<OT>::SolType& branching_costs,
		typename Solver<OT>::SolContainer& lb, typename Solver<OT>::SolContainer& left_lower_bound, typename Solver<OT>::SolContainer& right_lower_bound,
		ADataView& left_data, const Branch& left_branch, int left_depth, int left_nodes,
		ADataView& right_data, const Branch& right_branch, int right_depth, int right_nodes) {
		left_lower_bound = InitializeSol<OT>(true);
		right_lower_bound = InitializeSol<OT>(true);
		lb = InitializeSol<OT>(true);
		if (solver_parameters.use_lower_bounding) {
			right_lower_bound = cache->RetrieveLowerBound(right_data, right_branch, right_depth, right_nodes);
			//if (right_lower_bound->Size() > 0) stats.num_cache_hit_nonzero_bound++;
			left_lower_bound = cache->RetrieveLowerBound(left_data, left_branch, left_depth, left_nodes);
			//if (left_lower_bound->Size() > 0) stats.num_cache_hit_nonzero_bound++;

			if constexpr (OT::total_order) {
				CombineSols(feature, left_lower_bound, right_lower_bound, branching_costs, lb);
			} else {
				LBMerge(feature, context, left_lower_bound, right_lower_bound, branching_costs, lb);
			}
		}
	}

	template<class OT>
	template <bool reconstruct, typename U, typename std::enable_if<!U::total_order, int>::type>
	void Solver<OT>::Merge(int feature, const Solver<OT>::Context& context, typename Solver<OT>::SolContainer& UB,
		typename Solver<OT>::SolContainer& left_sols, typename Solver<OT>::SolContainer& right_sols, const typename Solver<OT>::SolType& branching_costs, typename Solver<OT>::SolContainer& final_sols, TreeNode<U>* tree) {
		runtime_assert(reconstruct != (tree == nullptr));
		if (left_sols->Size() == 0 || right_sols->Size() == 0) return;
		clock_t clock_start = clock();
		const int current_depth = context.GetBranch().Depth();

		// Compute new solutions by computing the product of left and right solutions
		Node<OT> node;
		for (auto& sol_left : left_sols->GetSolutions()) {
			for (auto& sol_right : right_sols->GetSolutions()) {
				if constexpr (reconstruct) {
					if (CheckReconstructSolution<OT>(sol_left, sol_right, branching_costs, tree)) return;
				} else {
					CombineSols(feature, sol_left, sol_right, branching_costs, node);
					if (!SatisfiesConstraint(node, context)) continue;
					if (solver_parameters.use_upper_bounding && UB->StrictDominates(node)) continue;
					AddSol<OT>(task, current_depth, final_sols, node);
					UpdateUB(context, UB, node);
				}
			}
		}

		stats.time_merging += double(clock() - clock_start) / CLOCKS_PER_SEC;
	}

	template<class OT>
	template <typename U, typename std::enable_if<!U::total_order, int>::type>
	void Solver<OT>::LBMerge(int feature, const Solver<OT>::Context& context, typename Solver<OT>::SolContainer& left_sols, typename Solver<OT>::SolContainer& right_sols, const typename Solver<OT>::SolType& branching_costs, typename Solver<OT>::SolContainer& final_lb) {
		if (left_sols->Size() == 0 || right_sols->Size() == 0) return;
		clock_t clock_start = clock();
		const Branch& branch = context.GetBranch();
		const int current_depth = branch.Depth();
		const size_t MAX_SIZE = solver_parameters.UB_LB_max_size;
		Container<U> small_left_sols;
		Container<U> small_right_sols;
		Container<U>* left_sols_ptr = left_sols.get();
		Container<U>* right_sols_ptr = right_sols.get();

		// To prevent high computation time, reduce the left and right LB to at most MAX_SIZE items
		if (left_sols->Size() > MAX_SIZE) {
			small_left_sols.AddInvOrMerge(*(left_sols.get()), MAX_SIZE);
			left_sols_ptr = &small_left_sols;
		}
		if (right_sols->Size() > MAX_SIZE) {
			small_right_sols.AddInvOrMerge(*(right_sols.get()), MAX_SIZE);
			right_sols_ptr = &small_right_sols;
		}

		// Combine left and right solutions and compute a new LB
		Node<OT> node;
		for (auto& sol_left : left_sols_ptr->GetSolutions()) {
			for (auto& sol_right : right_sols_ptr->GetSolutions()) {
				CombineSols(feature, sol_left, sol_right, branching_costs, node);
				AddSol<OT>(final_lb, node);
			}
		}
		stats.time_lb_merging += double(clock() - clock_start) / CLOCKS_PER_SEC;
	}

	template <class OT>
	void Solver<OT>::SubtractUBs(const Solver<OT>::Context& context, const typename Solver<OT>::SolContainer& UB, const typename Solver<OT>::SolContainer& sols,
			const typename Solver<OT>::SolContainer& current_solutions, const typename Solver<OT>::SolType& branching_costs, typename Solver<OT>::SolContainer& updatedUB) {
		clock_t clock_start = clock();
		const Branch& branch = context.GetBranch();
		if constexpr (OT::total_order) {
			if (solver_parameters.use_upper_bounding && solver_parameters.subtract_ub) {
				// Subtract the solution and the branching costs
				if (LeftDominatesRight<OT>(current_solutions.solution - OT::minimum_difference, UB.solution)) {
					OT::Subtract(current_solutions.solution - OT::minimum_difference, sols.solution, updatedUB.solution);
				} else {
					OT::Subtract(UB.solution, sols.solution, updatedUB.solution);
				}
				OT::Subtract(updatedUB.solution, branching_costs, updatedUB.solution);
			} else {
				updatedUB.solution = UB.solution;
			}
		} else {
			const size_t MAX_SIZE = solver_parameters.UB_LB_max_size;
			if (!solver_parameters.use_upper_bounding || !solver_parameters.subtract_ub || sols->Size() == 0) {
				updatedUB = CopySol<OT>(UB);
				if (solver_parameters.use_upper_bounding) {
					// In the root nod of the search, feasible solutions can be relaxed by removing information that is related
					// to constraint satisfaction from the solution
					AddRootRelaxSols<OT>(task, branch, updatedUB, current_solutions);
				}
				return;
			}
			Node<OT> diffsol;

			// Reduce size of UB
			const Solver<OT>::SolContainer* UB_ptr = &UB;
			Solver<OT>::SolContainer small_UB = InitializeSol<OT>();
			if (UB->Size() >= MAX_SIZE) {
				small_UB = InitializeSol<OT>();
				small_UB->AddOrInvMerge(*(UB.get()), MAX_SIZE);
				UB_ptr = &small_UB;
			}

			// Subtract branching costs
			Solver<OT>::SolContainer branch_substract_UB = InitializeSol<OT>();
			if (OT::has_branching_costs) {
				for (size_t i = 0; i < (*UB_ptr)->Size(); i++) {
					OT::Subtract((*UB_ptr)->GetSolutions()[i].solution, branching_costs, diffsol.solution);
					branch_substract_UB->Add(diffsol);
				}
				UB_ptr = &branch_substract_UB;
			}

			// Reduce size of sols
			const Solver<OT>::SolContainer* sols_ptr = &sols;
			Solver<OT>::SolContainer small_sols = InitializeSol<OT>();
			if (sols->Size() >= MAX_SIZE) {
				small_sols->AddOrMerge(*(sols.get()), MAX_SIZE);
				sols_ptr = &small_sols;
			}

			Solver<OT>::SolContainer corner_union = InitializeSol<OT>();
			Solver<OT>::SolContainer sub_union = InitializeSol<OT>();
			auto extreme_points = OT::ExtremePoints();
			for (size_t j = 0; j < (*sols_ptr)->Size(); j++) {
				Solver<OT>::SolContainer sub_ub = InitializeSol<OT>();
				for (size_t i = 0; i < (*UB_ptr)->Size(); i++) {
					OT::Subtract((*UB_ptr)->Get(i).solution, (*sols_ptr)->Get(j).solution, diffsol.solution);
					//sub_ub->AddOrInvMerge(diffsol, MAX_SIZE);
					sub_ub->Add(diffsol);
				}
				for (auto& ep : extreme_points) {
					sub_ub->Add(Node<OT>(ep));
				}

				// Compute 'staircase corners'
				Solver<OT>::SolContainer corners = InitializeSol<OT>();
				for (size_t i = 0; i < sub_ub->Size(); i++) {
					for (size_t k = i + 1; k < sub_ub->Size(); k++) {
						OT::MergeInv(sub_ub->Get(i).solution, sub_ub->Get(k).solution, diffsol.solution);
						//corners->AddOrInvMerge(diffsol, MAX_SIZE);
						corners->Add(diffsol);
					}
				}
				//corner_union->AddInvOrInvMerge(*(corners.get()), MAX_SIZE);
				//sub_union->AddInvOrInvMerge(*(sub_ub.get()), MAX_SIZE);
				
				corner_union->AddInv(*(corners.get()));
				sub_union->AddInv(*(sub_ub.get()));
			}
			// Compute 'staircase corners'
			for (auto& ep : extreme_points) {
				corner_union->Add(Node<OT>(ep));
			}
			for (size_t i = 0; i < corner_union->Size(); i++) {
				for (size_t k = i + 1; k < corner_union->Size(); k++) {
					OT::Merge(corner_union->Get(i).solution, corner_union->Get(k).solution, diffsol.solution);
					updatedUB->AddInvOrInvMerge(diffsol, MAX_SIZE);
				}
			}
			updatedUB->AddInvOrInvMerge(*(sub_union.get()), MAX_SIZE);
			//updatedUB->AddOrInvMerge(*(UB_ptr->get()), MAX_SIZE);
			
		}
		// In the root nod of the search, feasible solutions can be relaxed by removing information that is related
		// to constraint satisfaction from the solution
		AddRootRelaxSols<OT>(task, branch, updatedUB, current_solutions);
		stats.time_ub_subtracting += double(clock() - clock_start) / CLOCKS_PER_SEC;
	}

	template<class OT>
	bool Solver<OT>::SatisfiesConstraint(const Node<OT>& sol, const Solver<OT>::Context& context) const {
		if constexpr (!OT::has_constraint) {
			return true;
		} else {
			return task->SatisfiesConstraint(sol, context);
		}
	}

	template <class OT>
	void Solver<OT>::UpdateUB(const Solver<OT>::Context& context, typename Solver<OT>::SolContainer& UB, Node<OT> sol) const {
		if (solver_parameters.use_upper_bounding) {
			AddSol<OT>(UB, sol);
		}
	}

	template <class OT>
	void Solver<OT>::ReduceNodeBudget(const ADataView& data, const Solver<OT>::Context& context, const typename Solver<OT>::SolContainer& UB, int& max_depth, int& num_nodes) const {
		if constexpr (OT::total_order && OT::has_branching_costs && OT::constant_branching_costs &&
			std::is_same<typename OT::SolType, double>::value || std::is_same<typename OT::SolType, int>::value) {
			auto branching_costs = GetBranchingCosts(data, context, 0);
			if (branching_costs <= 0) return;
			int nodes = std::max(0, int((UB.solution + 1e-6) / branching_costs));
			if (nodes < num_nodes) {
				int new_max_depth = std::min(max_depth, nodes);
				if (new_max_depth < max_depth) {
					max_depth = new_max_depth;
					num_nodes = std::min(num_nodes, (1 << max_depth) - 1);
					runtime_assert(max_depth <= num_nodes)
				}
			}
		}
	}

	template <class OT>
	std::shared_ptr<SolverResult> Solver<OT>::HyperSolve(const ADataView& train_data) {
		using ScoreType = std::shared_ptr<Score>;
		runtime_assert(parameters.GetBooleanParameter("hyper-tune"));
		stopwatch.Initialise(parameters.GetFloatParameter("time"));

		bool verbose = parameters.GetBooleanParameter("verbose");
		const int max_num_nodes = int(parameters.GetIntegerParameter("max-num-nodes"));

		for (int tune_phase = 0; tune_phase < OT::num_tune_phases; tune_phase++) {
			int best_config = -1;
			double best_score;
			auto tuning_config = OT::GetTuneRunConfiguration(parameters, train_data, tune_phase);

			const int n_runs = tuning_config.GetNumberOfRuns();
			const int n_configs = tuning_config.GetNumberOfConfigs();
			const double validation_percentage = tuning_config.validation_percentage;

			std::vector<std::vector<ScoreType>> performances(n_configs, std::vector<ScoreType>(n_runs));

			std::vector<ADataView> sub_train_datas, sub_test_datas;
			ADataView::KFoldSplit<typename OT::LabelType>(train_data, sub_train_datas, sub_test_datas, rng, n_runs, false);

			for (int r = 0; r < n_runs; r++) {
				int ix = 0;
				Solver<OT> solver(parameters, rng);
				solver.solver_parameters.verbose = false;
				//ADataView::TrainTestSplitData<typename OT::LabelType>(train_data, sub_train_data, sub_test_data, rng, validation_percentage, true);
				ADataView& sub_train_data = sub_train_datas[r], &sub_test_data = sub_test_datas[r];
				solver.InitializeSolver(sub_train_data); // Initialize with max-depth
				solver.InitializeTest(sub_test_data);
				auto worst = InternalTestScore<OT>::GetWorst(solver.task);
				for (int c = 0; c < n_configs;c++) {
					if (!stopwatch.IsWithinTimeLimit()) {
						performances[c][r] = worst;
						continue;
					}
					if (verbose) {
						std::cout << "Tune phase " << (tune_phase + 1) << "/" << OT::num_tune_phases
							<< " Split " << (r + 1) << "/" << n_runs
							<< " Config " << (c + 1) << "/" << n_configs
							<< " \t" << tuning_config.descriptors[c];
					}
					
					bool reset = solver.parameters.GetIntegerParameter("max-depth") < tuning_config.parameters[c].GetIntegerParameter("max-depth");
					solver.parameters = tuning_config.parameters[c];
					solver.parameters.SetFloatParameter("time", stopwatch.TimeLeftInSeconds());
					solver.InitializeSolver(sub_train_data, tuning_config.reset_solver || reset);
					const auto result = solver.Solve(sub_train_data);
					const auto test_result = solver.TestPerformance(result, sub_test_data);					
					if (result->NumSolutions() == 0 || !result->IsProvenOptimal()) {
						if (c > 0) performances[c][r] = performances[c - 1][r];
						else performances[c][r] = worst;
					} else {
						performances[c][r] = test_result->scores[test_result->best_index];
					}
					if(verbose) std::cout << " \tScore: " << OT::ScoreToString(performances[c][r]->score) << std::endl;
					if (tuning_config.skip_when_max_tree && result->GetBestNodeCount() == max_num_nodes && c + 1 < n_configs) {
						if (verbose) std::cout << "Reached maximum tree. Skipping configuration " << (c + 2) << " to " << n_configs << std::endl;
						for (c++; c < n_configs;c++) {
							performances[c][r] = performances[c - 1][r];
						}
					}
				}
			}

			for (int c = 0; c < n_configs; c++) {
				auto avg_perf = InternalTestScore<OT>::GetAverage(performances[c]);
				auto& score = avg_perf->score;
				if (best_config == -1 || OT::CompareScore(score, best_score)) {
					best_config = c;
					best_score = score;
				}
			}

			if (verbose) {
				std::cout << std::endl << "Finished hyper parameter search (phase " << (tune_phase+1) << "/" << OT::num_tune_phases << "). Best config : " <<  tuning_config.descriptors[best_config] << std::endl << std::endl;
			}
			parameters = tuning_config.parameters[best_config];
		}
		stats.total_time += stopwatch.TimeElapsedInSeconds();
		parameters.SetFloatParameter("time", stopwatch.TimeLeftInSeconds());
		return Solve(train_data);
	}


	template<class OT>
	std::shared_ptr<Tree<OT>> Solver<OT>::ConstructOptimalTree(const Node<OT>& sol, ADataView& data, const Solver<OT>::Context& context, int max_depth, int num_nodes) {
		runtime_assert(num_nodes >= 0);
		max_depth = std::min(max_depth, num_nodes);

		if (max_depth == 0 || num_nodes == 0 || sol.NumNodes() == 0) {
			return Tree<OT>::CreateLabelNode(sol.label);
		}

		// Special D1 reconstruct
		bool d1 = max_depth == 1 || num_nodes == 1 || sol.NumNodes() == 1;
		bool use_cache = cache->UseCache();

		// Special D2 reconstruct
		if constexpr (OT::use_terminal) {
			if (!d1 && IsTerminalNode(max_depth, num_nodes)) {
				try {
					return terminal_solver1->ConstructOptimalTree(sol, data, context, max_depth, num_nodes);
				} catch (std::exception& e) {
					// Could happen because of numerical instability, but could also refer to an actual error. 
					// Check if the solution value is the same when the terminal-solver is not used.
					use_cache = false; 
				}
			}
		}

		// Initialize empty UBs
		auto UB = InitializeSol<OT>();
		AddSol<OT>(UB, OT::worst);
		auto UBleft = InitializeSol<OT>();
		AddSol<OT>(UBleft, OT::worst);
		auto UBright = InitializeSol<OT>();
		AddSol<OT>(UBright, OT::worst);

		auto tree = Tree<OT>::CreateFeatureNodeWithNullChildren(sol.feature);

		const Branch& branch = context.GetBranch();
		ADataView left_data, right_data;
		data_splitter.Split(data, branch, sol.feature, left_data, right_data);
		runtime_assert(left_data.Size() >= solver_parameters.minimum_leaf_node_size && right_data.Size() >= solver_parameters.minimum_leaf_node_size);
		Solver<OT>::Context left_context, right_context;
		task->GetLeftContext(data, context, sol.feature, left_context);
		task->GetRightContext(data, context, sol.feature, right_context);

		const int left_subtree_size = sol.num_nodes_left;
		const int right_subtree_size = sol.num_nodes_right;
		int left_depth = std::min(max_depth - 1, left_subtree_size);
		int right_depth = std::min(max_depth - 1, right_subtree_size);

		int left_size = left_subtree_size;
		int right_size = right_subtree_size;
		Solver<OT>::SolContainer left_sols, right_sols;
		
		if (use_cache) {
			const int max_size_subtree = std::min((1 << (max_depth - 1)) - 1, num_nodes - 1); //take the minimum between a full tree of max_depth or the number of nodes - 1
			const int min_size_subtree = num_nodes - 1 - max_size_subtree;
			int min_left_subtree_size = std::max(sol.num_nodes_left, min_size_subtree);
			int min_right_subtree_size = std::max(sol.num_nodes_right, min_size_subtree);

			left_size = min_left_subtree_size;
			for (; left_size <= max_size_subtree; left_size++) {
				left_depth = std::min(max_depth - 1, left_size);
				if (left_size == 0)
					left_sols = SolveLeafNode(left_data, left_context, UBleft);
				else
					left_sols = cache->RetrieveOptimalAssignment(left_data, left_context.GetBranch(), left_depth, left_size);
				if (!CheckEmptySol<OT>(left_sols)) break;
			}
			if constexpr (!OT::total_order) left_sols->FilterOnNumberOfNodes(min_left_subtree_size);

			right_size = min_right_subtree_size;
			for (; right_size <= max_size_subtree; right_size++) {
				right_depth = std::min(max_depth - 1, right_size);
				if (right_size == 0)
					right_sols = SolveLeafNode(right_data, right_context, UBright);
				else
					right_sols = cache->RetrieveOptimalAssignment(right_data, right_context.GetBranch(), right_depth, right_size);
				if (!CheckEmptySol<OT>(right_sols)) break;
			}
			if constexpr (!OT::total_order) right_sols->FilterOnNumberOfNodes(min_right_subtree_size);

		}
		
		if (!use_cache || CheckEmptySol<OT>(left_sols)) {
			left_depth = std::min(max_depth - 1, left_subtree_size);
			left_sols = SolveSubTree(left_data, left_context, UBleft, left_depth, left_subtree_size);
			if constexpr (!OT::total_order) left_sols->FilterOnNumberOfNodes(left_subtree_size);
		}
		if (!use_cache || CheckEmptySol<OT>(right_sols)) {
			right_depth = std::min(max_depth - 1, right_subtree_size);
			right_sols = SolveSubTree(right_data, right_context, UBright, right_depth, right_subtree_size);
			if constexpr (!OT::total_order) right_sols->FilterOnNumberOfNodes(right_subtree_size);
		}

		// Reconstruct Merge
		if constexpr (!OT::total_order) {
			TreeNode<OT> tree_node;
			tree_node.parent = sol;
			auto branching_costs = GetBranchingCosts(data, context, sol.feature);
			auto empty_final_sols = std::shared_ptr<Container<OT>>(nullptr);
			Merge<true>(sol.feature, context, UB, left_sols, right_sols, branching_costs, empty_final_sols, &tree_node);

			runtime_assert(tree_node.left_child.IsFeasible());
			runtime_assert(tree_node.right_child.IsFeasible());

			tree->left_child = ConstructOptimalTree(tree_node.left_child, left_data, left_context, left_depth, left_size);
			tree->right_child = ConstructOptimalTree(tree_node.right_child, right_data, right_context, right_depth, right_size);


		} else {
			tree->left_child = ConstructOptimalTree(left_sols, left_data, left_context, left_depth, left_size);
			tree->right_child = ConstructOptimalTree(right_sols, right_data, right_context, right_depth, right_size);
		}		
		
		return tree;
	}

	template <class OT>
	template <typename U, typename std::enable_if<U::element_additive, int>::type>
	bool Solver<OT>::UpdateCacheUsingSimilarity(ADataView& data, const Branch& branch, int max_depth, int num_nodes) {
		PairLowerBoundOptimal<OT> result = similarity_lower_bound_computer->ComputeLowerBound(data, branch, max_depth, num_nodes, cache);
		if (CheckEmptySol<OT>(result.lower_bound)) return false;
		if (result.optimal) { return true; }
		static SolContainer empty_sol = InitializeSol<OT>(true);
		if (!SolutionsEqual<OT>(empty_sol, result.lower_bound)) {
			cache->UpdateLowerBound(data, branch, result.lower_bound, max_depth, num_nodes);
		}
		return false;

	}

	template <class OT>
	typename Solver<OT>::SolType Solver<OT>::GetBranchingCosts(const ADataView& data, const Solver<OT>::Context& context, int feature) const {
		if constexpr (!OT::has_branching_costs) {
			return OT::best;
		} else {
			return task->GetBranchingCosts(data, context, feature);
		}
	}

	template <class OT>
	void Solver<OT>::PreprocessData(AData& data, bool train) {
		if (train) {
			flipped_features.resize(data.NumFeatures(), 0);
			for (int f = 0; f < data.NumFeatures(); f++) {
				int positive_count = 0;
				for (int i = 0; i < data.Size(); i++) {
					auto instance = data.GetInstance(i);
					if (instance->IsFeaturePresent(f))
						positive_count++;
				}
				if (positive_count > data.Size() / 2) {
					// Flip this feature, to improve the performance of the D2-solver
					flipped_features[f] = 1;
					for (int i = 0; i < data.Size(); i++) {
						auto instance = data.GetMutableInstance(i);
						instance->FlipFeature(f);
					}
				}
			}
		} else {
			for (int f = 0; f < data.NumFeatures(); f++) {
				if (flipped_features[f] == 1) {
					for (int i = 0; i < data.Size(); i++) {
						auto instance = data.GetMutableInstance(i);
						instance->FlipFeature(f);
					}
				}
			}
		}
		if constexpr (OT::preprocess_data) {
			task->PreprocessData(data, train);
		}
	}

	template <class OT>
	void Solver<OT>::PreprocessTrainData(const ADataView& org_train_data, ADataView& train_data) {
		train_data = org_train_data;
		if constexpr (OT::preprocess_train_test_data) {
			task->PreprocessTrainData(train_data);
		}
	}

	template <class OT>
	void Solver<OT>::PreprocessTestData(const ADataView& org_test_data, ADataView& test_data) {
		test_data = org_test_data;
		if constexpr (OT::preprocess_train_test_data) {
			task->PreprocessTestData(test_data);
		}
	}

	template <class OT>
	std::shared_ptr<SolverResult> Solver<OT>::TestPerformance(const std::shared_ptr<SolverResult>& _result, const ADataView& _test_data) {
		InitializeTest(_test_data, false);
		const SolverTaskResult<OT>* result = static_cast<const SolverTaskResult<OT>*>(_result.get());
		auto solver_result = std::make_shared<SolverTaskResult<OT>>(*result);
		for (size_t i = 0; i < result->NumSolutions(); i++) {
			auto score = InternalTestScore<OT>::ComputeTestPerformance(&data_splitter, task, result->trees[i].get(), flipped_features, test_data);
			solver_result->SetScore(i, score);
		}
		return solver_result;
	}

	template <class OT>
	std::vector<typename OT::LabelType> Solver<OT>::Predict(const std::shared_ptr<Tree<OT>>& tree, const ADataView& _test_data) {
		InitializeTest(_test_data, false);
		std::vector<typename OT::LabelType> labels(test_data.Size());
		typename OT::ContextType context;
		tree->Classify(&data_splitter, task, context, flipped_features, test_data, labels);
		return labels;
	}

	template class Solver<Accuracy>;
	template class Solver<CostComplexAccuracy>;

	template class Solver<CostSensitive>;
	template class Solver<F1Score>;
	template class Solver<GroupFairness>;
	template class Solver<EqOpp>;
	template class Solver<PrescriptivePolicy>;
	template class Solver<SurvivalAnalysis>;
}

