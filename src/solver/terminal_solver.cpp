#include "solver/terminal_solver.h"
#include "solver/solver.h"

namespace STreeD {

	template <class OT>
	TerminalSolver<OT>::TerminalSolver(Solver<OT>* solver) :
		task(solver->GetTask()), num_features(solver->NumFeatures()), 
		cost_calculator(solver->GetTask(), solver->NumFeatures(), solver->NumLabels()),
		children_info(solver->NumFeatures()), solver_parameters(&(solver->GetSolverParameters())),
		temp_branch_node(INT32_MAX, OT::worst, 0, 0),
		temp_leaf_node(OT::worst_label, OT::worst),
		sols(solver->NumLabels())
	{
		num_labels = solver->NumLabels();
		for (int left_label = 0; left_label < num_labels; left_label++) {
			for (int right_label = 0; right_label < num_labels; right_label++) {
				if (num_labels > 1 && left_label == right_label) continue;
				label_assignments.push_back({ left_label, right_label });
			}
		}
	}

	template <class OT>
	TerminalResults<OT>& TerminalSolver<OT>::Solve(const ADataView& data, const typename TerminalSolver<OT>::Context& context, typename TerminalSolver<OT>::SolContainer& UB, int num_nodes) {
		this->UB = UB;
		bool changes_made = cost_calculator.Initialize(data, context, num_nodes);
		if (!changes_made) return results;
		results.Clear();
		if (cost_calculator.GetTotalCount() < solver_parameters->minimum_leaf_node_size) return results;
		InitialiseChildrenInfo(context, data);
		SolveOneNode(data, context, true);
		if (num_nodes == 1) {
			return results;
		}

		typename TerminalSolver<OT>::SolType left_sol;
		typename TerminalSolver<OT>::SolType right_sol;
		int  total0, total1;
		IndexInfo index;
		Counts counts;
		for (int f1 = 0; f1 < num_features; f1++) {
			if (!task->MayBranchOnFeature(f1)) continue;
			total0 = cost_calculator.GetCount00(f1, f1);
			total1 = cost_calculator.GetCount11(f1, f1);
			if (total0 < solver_parameters->minimum_leaf_node_size || total1 < solver_parameters->minimum_leaf_node_size) continue;

			for (int f2 = f1 + 1; f2 < num_features; f2++) {
				if (!task->MayBranchOnFeature(f2)) continue;
				if (f1 == f2) continue;

				cost_calculator.GetIndexInfo(f1, f2, index);
				cost_calculator.GetCounts(counts, index);
				runtime_assert(total0 == counts.count00 + counts.count01);
				runtime_assert(total1 == counts.count10 + counts.count11);

				if ((counts.count00 < solver_parameters->minimum_leaf_node_size || counts.count01 < solver_parameters->minimum_leaf_node_size)
					&& (counts.count10 < solver_parameters->minimum_leaf_node_size || counts.count11 < solver_parameters->minimum_leaf_node_size)
					&& (counts.count00 < solver_parameters->minimum_leaf_node_size || counts.count10 < solver_parameters->minimum_leaf_node_size)
					&& (counts.count01 < solver_parameters->minimum_leaf_node_size || counts.count11 < solver_parameters->minimum_leaf_node_size)) {
					continue;
				}

				const auto branch_left_costs = cost_calculator.GetBranchingCosts0(counts.count00  + counts.count01, f1, f2);
				const auto branch_right_costs = cost_calculator.GetBranchingCosts1(counts.count10 + counts.count11 , f1, f2);

				const auto branch_left_costs_rev = cost_calculator.GetBranchingCosts0(counts.count00  + counts.count10 , f2, f1);
				const auto branch_right_costs_rev = cost_calculator.GetBranchingCosts1(counts.count01 + counts.count11, f2, f1);



				for (int label = 0; label < num_labels; label++) {
					cost_calculator.CalcSols(counts, sols[label], label, index);
				}

				// Find best left child (first=0)
				if (counts.count00 >= solver_parameters->minimum_leaf_node_size && counts.count01 >= solver_parameters->minimum_leaf_node_size) {
					for (auto& la : label_assignments) {
						OT::Add(sols[la.left_label].sol00, sols[la.right_label].sol01, left_sol);
						OT::Add(left_sol, branch_left_costs, left_sol);
						UpdateBestLeftChild(f1, f2, left_sol);
					}
				}
				
				// Find best right child (first=1)
				if (counts.count10 >= solver_parameters->minimum_leaf_node_size && counts.count11 >= solver_parameters->minimum_leaf_node_size) {
					for (auto& la : label_assignments) {
						OT::Add(sols[la.left_label].sol10, sols[la.right_label].sol11, right_sol);
						OT::Add(right_sol, branch_right_costs, right_sol);
						UpdateBestRightChild(f1, f2, right_sol);
					}
				}

				// Find best left child (rev, first=0)
				if (counts.count00 >= solver_parameters->minimum_leaf_node_size && counts.count10 >= solver_parameters->minimum_leaf_node_size) {
					for (auto& la : label_assignments) {
						OT::Add(sols[la.left_label].sol00, sols[la.right_label].sol10, left_sol);
						OT::Add(left_sol, branch_left_costs_rev, left_sol);
						UpdateBestLeftChild(f2, f1, left_sol);
					}
				}

				// Find best right child (rev, first=1)
				if (counts.count01 >= solver_parameters->minimum_leaf_node_size && counts.count11 >= solver_parameters->minimum_leaf_node_size) {
					for (auto& la : label_assignments) {
						OT::Add(sols[la.left_label].sol01, sols[la.right_label].sol11, right_sol);
						OT::Add(right_sol, branch_right_costs_rev, right_sol);
						UpdateBestRightChild(f2, f1, right_sol);
					}
				}

			}

			UpdateBestTwoNodeAssignment(context, f1);
			UpdateBestThreeNodeAssignment(context, f1);

		}

		AddSols<OT>(results.two_nodes_solutions, results.one_node_solutions);
		AddSols<OT>(results.three_nodes_solutions, results.two_nodes_solutions);
		return results;
	}

	template <class OT>
	void TerminalSolver<OT>::SolveOneNode(const ADataView& data, const typename TerminalSolver<OT>::Context& context, bool initialized) {
		runtime_assert(initialized); // for now
		auto& result = results.one_node_solutions;
		SetSolSizeBudget<OT>(result, 1, 1);

		Node<OT> node;
		typename TerminalSolver<OT>::SolType merged_sol;
		{
			typename OT::SolLabelType out_label;
			for (int label = 0; label < data.NumLabels(); label++) {
				cost_calculator.CalcLeafSol(merged_sol, label, out_label);
				node.Set(INT32_MAX, out_label, merged_sol, 0, 0);

				if (OT::has_constraint && !SatisfiesConstraint(node, context)) continue;
				if (OT::terminal_filter && LeftStrictDominatesRightSol<OT>(UB, node)) continue;
				AddSol<OT>(result, node);
			}
		}
		
		bool computed_leaves = false;

		if (initialized) {
			Counts counts;
			IndexInfo index;
			for (int feature = 0; feature < num_features; feature++) {
				if (!task->MayBranchOnFeature(feature)) continue;
				cost_calculator.GetIndexInfo(feature, feature, index);
				cost_calculator.GetCounts(counts, index);
				runtime_assert(counts.count00 + counts.count11 >= solver_parameters->minimum_leaf_node_size); // If even a leaf node is too small, D2-solver should not be called
				if (counts.count00 < solver_parameters->minimum_leaf_node_size || counts.count11 < solver_parameters->minimum_leaf_node_size) continue;
				for (int label = 0; label < num_labels; label++) {
					cost_calculator.CalcSols(counts, sols[label], label, index);
				}
				auto branching_costs = cost_calculator.GetBranchingCosts(feature);

				//for every possible combination of different left and right labels
				for (auto& la : label_assignments) {
					OT::Add(sols[la.left_label].sol00, sols[la.right_label].sol11, merged_sol);
					OT::Add(merged_sol, branching_costs, merged_sol);
					node.Set(feature, OT::worst_label, merged_sol, 0, 0);
					if (OT::has_constraint && !SatisfiesConstraint(node, context)) continue;
					if (OT::terminal_filter && LeftStrictDominatesRightSol<OT>(UB, node)) continue;
					AddSol<OT>(result, node);
					AddSol<OT>(UB, node);
				}
			}
		}
	}

	template <class OT>
	void TerminalSolver<OT>::InitialiseChildrenInfo(const Context& context, const ADataView& data) {
		for (int f = 0; f < num_features; f++) {
			auto& child_info = children_info[f];
			child_info.Clear();
			if constexpr (OT::terminal_compute_context) {
				task->GetLeftContext(data, context, f, child_info.left_context);
				task->GetRightContext(data, context, f, child_info.right_context);
			}
		}
	}

	template <class OT>
	void TerminalSolver<OT>::UpdateBestLeftChild(int root_feature, int feature, const SolType& solution) {
		auto& child_info = children_info[root_feature];
		const auto & context = child_info.left_context;
		temp_branch_node.feature = feature;
		temp_branch_node.solution = solution;
		if (OT::has_constraint && !SatisfiesConstraint(temp_branch_node, context)) return;
		if (OT::terminal_filter && LeftStrictDominatesRightSol<OT>(UB, temp_branch_node)) return;
		AddSol<OT>(child_info.left_child_assignments, temp_branch_node);
	}

	template <class OT>
	void TerminalSolver<OT>::UpdateBestRightChild(int root_feature, int feature, const SolType& solution) {
		auto& child_info = children_info[root_feature];
		const auto& context = child_info.right_context;
		temp_branch_node.feature = feature;
		temp_branch_node.solution = solution;
		if (OT::has_constraint && !SatisfiesConstraint(temp_branch_node, context)) return;
		if (OT::terminal_filter && LeftStrictDominatesRightSol<OT>(UB, temp_branch_node)) return;
		AddSol<OT>(child_info.right_child_assignments, temp_branch_node);
	}

	template <class OT>
	void TerminalSolver<OT>::UpdateBestTwoNodeAssignment(const typename TerminalSolver<OT>::Context& context, int root_feature) {
		auto& child_info = children_info[root_feature];
		const auto& left_context = child_info.left_context;
		const auto& right_context = child_info.right_context;
		Counts counts;
		IndexInfo index;
		
		auto left_leaves = InitializeSol<OT>();
		auto right_leaves = InitializeSol<OT>();

		cost_calculator.GetIndexInfo(root_feature, root_feature, index);
		cost_calculator.GetCounts(counts, index);
		int left_size = counts.count00;
		int right_size = counts.count11;

		typename OT::SolD2Type costs;
		typename TerminalSolver<OT>::SolType leaf_sol;
		typename OT::SolLabelType assign_label;

		Node<OT> node;
		if (left_size >= solver_parameters->minimum_leaf_node_size) {
			for (int label = 0; label < num_labels; label++) {
				costs = cost_calculator.GetCosts00(label, root_feature, root_feature);
				task->ComputeD2Costs(costs, left_size, leaf_sol);
				assign_label = cost_calculator.GetLabel(label, costs, left_size);
				node.Set(INT32_MAX, assign_label, leaf_sol, 0, 0);
				//node.Set(INT32_MAX, OT::worst_label, sols[label].sol00, 0, 0);
				if (OT::has_constraint && !SatisfiesConstraint(node, left_context)) continue;
				if (OT::terminal_filter && LeftStrictDominatesRightSol<OT>(UB, node)) continue;
				AddSol<OT>(left_leaves, node);
			}
		}
		if (right_size >= solver_parameters->minimum_leaf_node_size) {
			for (int label = 0; label < num_labels; label++) {
				costs = cost_calculator.GetCosts11(label, root_feature, root_feature);
				task->ComputeD2Costs(costs, right_size, leaf_sol);
				assign_label = cost_calculator.GetLabel(label, costs, right_size);
				node.Set(INT32_MAX, assign_label, leaf_sol, 0, 0);
				//node.Set(INT32_MAX, OT::worst_label, sols[label].sol11, 0, 0);
				if (OT::has_constraint && !SatisfiesConstraint(node, right_context)) continue;
				if (OT::terminal_filter && LeftStrictDominatesRightSol<OT>(UB, node)) continue;
				AddSol<OT>(right_leaves, node);
			}
		}

		auto left_children = children_info[root_feature].left_child_assignments;
		auto right_children = children_info[root_feature].right_child_assignments;

		if constexpr (!OT::total_order) {
			Merge(root_feature, context, left_children, right_leaves);
			Merge(root_feature, context, left_leaves, right_children);
		} else {
			Node<OT> new_node;
			auto branching_costs = cost_calculator.GetBranchingCosts(root_feature);
			if (!CheckEmptySol<OT>(left_children) && !CheckEmptySol<OT>(right_leaves)) {
				CombineSols(root_feature, left_children, right_leaves, branching_costs, new_node);
				runtime_assert(new_node.solution >= -1e-6);
				if (!OT::has_constraint || SatisfiesConstraint(new_node, context)) {
					if (!OT::terminal_filter || !LeftStrictDominatesRightSol<OT>(UB, new_node)) {
						if (new_node.solution < results.two_nodes_solutions.solution) results.two_nodes_solutions = new_node;
					}
				}
			}
			if (!CheckEmptySol<OT>(left_leaves) && !CheckEmptySol<OT>(right_children)) {
				CombineSols(root_feature, left_leaves, right_children, branching_costs, new_node);
				runtime_assert(new_node.solution >= -1e-6);
				if (!OT::has_constraint || SatisfiesConstraint(new_node, context)) {
					if (!OT::terminal_filter || !LeftStrictDominatesRightSol<OT>(UB, new_node)) {
						if (new_node.solution < results.two_nodes_solutions.solution) results.two_nodes_solutions = new_node;
					}
				}
			}
		}

	}

	template <class OT>
	void TerminalSolver<OT>::UpdateBestThreeNodeAssignment(const typename TerminalSolver<OT>::Context& context, int root_feature) {
		auto left_children = children_info[root_feature].left_child_assignments;
		auto right_children = children_info[root_feature].right_child_assignments;
		if constexpr (!OT::total_order) {
			Merge(root_feature, context, left_children, right_children);
		} else {
			if (!CheckEmptySol<OT>(left_children) && !CheckEmptySol<OT>(right_children)) {
				auto branching_costs = cost_calculator.GetBranchingCosts(root_feature);
				Node<OT> new_node;
				CombineSols(root_feature, left_children, right_children, branching_costs, new_node);
				runtime_assert(new_node.solution >= -1e-6);
				if (!SatisfiesConstraint(new_node, context)) return;
				if (OT::terminal_filter && LeftStrictDominatesRightSol<OT>(UB, new_node)) return;
				if (new_node.solution < results.three_nodes_solutions.solution) results.three_nodes_solutions = new_node;
			}
		}

	}

	template<class OT>
	template <typename U, typename std::enable_if<!U::total_order, int>::type>
	void TerminalSolver<OT>::Merge(int feature, const typename TerminalSolver<OT>::Context& context, std::shared_ptr<Container<U>> left_solutions, std::shared_ptr<Container<U>> right_solutions) {
		if (left_solutions->Size() == 0 || right_solutions->Size() == 0) return;
		auto branching_costs = cost_calculator.GetBranchingCosts(feature);
		{
			Node<OT> node;
			for (auto& left_sol : left_solutions->GetSolutions()) {
				for (auto&  right_sol: right_solutions->GetSolutions()) {
					int nodes = left_sol.NumNodes() + right_sol.NumNodes() + 1;

					CombineSols(feature, left_sol, right_sol, branching_costs, node);
					if (!SatisfiesConstraint(node, context)) continue;
					if (OT::terminal_filter && LeftStrictDominatesRightSol<OT>(UB, node)) continue;
					if (nodes == 2) {
						results.two_nodes_solutions->Add(node);
					} else if (nodes == 3) {
						results.three_nodes_solutions->Add(node);
					}
				}
			}
		}
	}

	template<class OT>
	bool TerminalSolver<OT>::SatisfiesConstraint(const Node<OT>& sol, const TerminalSolver<OT>::Context& context) const {
		if constexpr (!OT::has_constraint || !OT::terminal_filter) {
			return true;
		} else {
			return task->SatisfiesConstraint(sol, context);
		}
	}

	template <class OT>
	std::shared_ptr<Tree<OT>> TerminalSolver<OT>::ConstructOptimalTree(const Node<OT>& node, const ADataView& data, const typename TerminalSolver<OT>::Context& context, int max_depth, int num_nodes) {
		runtime_assert(max_depth > 0 && num_nodes > 0);
		cost_calculator.InitializeReconstruct(data, context, node.feature);
		std::vector<TreeNode<OT>> left_solutions;
		std::vector<TreeNode<OT>> right_solutions;
		TreeNode<OT> left_solution, right_solution, tree_node;
		Node<OT> left_node, right_node;

		Counts counts;
		cost_calculator.GetCounts(counts, node.feature, node.feature);

		for (int label = 0; label < num_labels; label++) {
			cost_calculator.CalcSols(counts, sols[label], label, node.feature, node.feature);
		}
		if (node.num_nodes_left == 0) {
			for (int left_label = 0; left_label < num_labels; left_label++) {
				auto assign_label = cost_calculator.GetLabel00(left_label, node.feature, node.feature);
				temp_leaf_node.label = assign_label;
				temp_leaf_node.solution = sols[left_label].sol00;
				if (LeftStrictDominatesRight<OT>(node.solution, temp_leaf_node.solution)) continue;
				if constexpr (OT::total_order) {
					if (temp_leaf_node.solution < left_solution.parent.solution) {
						left_solution.parent = temp_leaf_node;
					}
				} else {
					left_solution.parent = temp_leaf_node;
					left_solutions.push_back(left_solution);
				}
			}
		}
		if (node.num_nodes_right == 0) {
			for (int right_label = 0; right_label < num_labels; right_label++) {
				auto assign_label = cost_calculator.GetLabel11(right_label, node.feature, node.feature);
				temp_leaf_node.label = assign_label;
				temp_leaf_node.solution = sols[right_label].sol11;
				if (LeftStrictDominatesRight<OT>(node.solution, temp_leaf_node.solution)) continue;
				if constexpr (OT::total_order) {
					if (temp_leaf_node.solution < right_solution.parent.solution) {
						right_solution.parent = temp_leaf_node;
					}
				} else {
					right_solution.parent = temp_leaf_node;
					right_solutions.push_back(right_solution);
				}
			}
		}
		Node<OT> temp_node;
		if (node.num_nodes_left > 0 || node.num_nodes_right > 0) {
			for (int f2 = 0; f2 < num_features; f2++) {
				if (f2 == node.feature) continue;

				cost_calculator.GetCounts(counts, node.feature, f2);
				
				for (int label = 0; label < num_labels; label++) {
					cost_calculator.CalcSols(counts, sols[label], label, node.feature, f2);
				}
				if (node.num_nodes_left > 0 && counts.count00 >= solver_parameters->minimum_leaf_node_size && counts.count01 >= solver_parameters->minimum_leaf_node_size) {
					auto branching_costs = cost_calculator.GetBranchingCosts0(counts.count00 + counts.count01, node.feature, f2);
					for (int left_label = 0; left_label < num_labels; left_label++) {
						for (int right_label = 0; right_label < num_labels; right_label++) {
							//if (num_labels > 1 && left_label == right_label) continue;
							auto left_assigned_label = cost_calculator.GetLabel00(left_label, node.feature, f2);
							auto right_assigned_label = cost_calculator.GetLabel01(right_label, node.feature, f2);
							left_node.Set(INT32_MAX, left_assigned_label, sols[left_label].sol00, 0, 0);
							right_node.Set(INT32_MAX, right_assigned_label, sols[right_label].sol01, 0, 0);
							CombineSols<OT>(f2, left_node, right_node, branching_costs, temp_node);
							if(LeftStrictDominatesRight<OT>(node.solution, temp_node.solution)) continue;
							if constexpr (OT::total_order) {
								if (temp_node.solution < left_solution.parent.solution) {
									left_solution.Set(temp_node, left_node, right_node);
								}
							} else {
								left_solution.Set(temp_node, left_node, right_node);
								left_solutions.push_back(left_solution);
							}
						}
					}
				}
				
				if (node.num_nodes_right > 0 && counts.count10 >= solver_parameters->minimum_leaf_node_size && counts.count11 >= solver_parameters->minimum_leaf_node_size) {
					auto branching_costs = cost_calculator.GetBranchingCosts1(counts.count10 + counts.count11, node.feature, f2);
					for (int left_label = 0; left_label < num_labels; left_label++) {
						for (int right_label = 0; right_label < num_labels; right_label++) {
							//if (num_labels > 1 && left_label == right_label) continue;
							auto left_assigned_label = cost_calculator.GetLabel10(left_label, node.feature, f2);
							auto right_assigned_label = cost_calculator.GetLabel11(right_label, node.feature, f2);
							left_node.Set(INT32_MAX, left_assigned_label, sols[left_label].sol10, 0, 0);
							right_node.Set(INT32_MAX, right_assigned_label, sols[right_label].sol11, 0, 0);
							CombineSols<OT>(f2, left_node, right_node, branching_costs, temp_node);
							if (LeftStrictDominatesRight<OT>(node.solution, temp_node.solution)) continue;
							if constexpr (OT::total_order) {
								if (temp_node.solution < right_solution.parent.solution) {
									right_solution.Set(temp_node, left_node, right_node);
								}
							} else {
								right_solution.Set(temp_node, left_node, right_node);
								right_solutions.push_back(right_solution);
							}
						}
					}
				}
			}
		}
		if constexpr (OT::total_order) {
			runtime_assert(left_solution.parent.IsFeasible());
			runtime_assert(right_solution.parent.IsFeasible());
			if (!left_solution.parent.IsFeasible() || !right_solution.parent.IsFeasible()) {
				throw std::runtime_error("Could not find a feasible tree for the given solution.");
			}
			tree_node.Set(node, left_solution.parent, right_solution.parent);
			return Tree<OT>::CreateD2TreeFromTreeNodes(tree_node, left_solution, right_solution);
		} else {
			auto branching_costs = cost_calculator.GetBranchingCosts(node.feature);
			tree_node.parent = node;
			for (const auto& left_sol : left_solutions) {
				for (const auto& right_sol : right_solutions) {
					if (CheckReconstructSolution<OT>(left_sol.parent, right_sol.parent, branching_costs, &tree_node)) {
						return Tree<OT>::CreateD2TreeFromTreeNodes(tree_node, left_sol, right_sol);
					}
				}
			}
			runtime_assert(1 == 0);
			throw std::runtime_error("Could not find a feasible tree for the given solution.");
		}
	}

	template class TerminalSolver<Accuracy>;
	template class TerminalSolver<CostComplexAccuracy>;

	template class TerminalSolver<Regression>;
	template class TerminalSolver<CostComplexRegression>;
	template class TerminalSolver<SimpleLinearRegression>;

	template class TerminalSolver<CostSensitive>;
	template class TerminalSolver<InstanceCostSensitive>;
	template class TerminalSolver<F1Score>;
	template class TerminalSolver<GroupFairness>;
	template class TerminalSolver<EqOpp>;
	template class TerminalSolver<PrescriptivePolicy>;
	template class TerminalSolver<SurvivalAnalysis>;

}