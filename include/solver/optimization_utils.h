#pragma once
#include "base.h"
#include "model/data.h"
#include "model/branch.h"
#include "model/node.h"
#include "model/container.h"


#define DBL_DIFF 1e-4

namespace STreeD {

	template <class OT>
	using SolContainer = typename std::conditional<OT::total_order, Node<OT>, std::shared_ptr<Container<OT>>>::type;

	/*
	* Initialize an empty solution. 
	* Default is a worst-case solution
	* if lb=true, it is a best-case solution
	*/
	template <class OT>
	SolContainer<OT> InitializeSol(bool lb = false) {
		if constexpr (OT::total_order) {
			if (lb) return Node<OT>(OT::best);
			return Node<OT>();
		} else {
			auto result = std::make_shared < Container<OT> >();
			if (lb) result->Add(OT::best);
			return result;
		}
	}

	/*
	* Initialize an empty lower bound.
	*/
	template <class OT>
	SolContainer<OT> InitializeLB() {
		return InitializeSol<OT>(true);
	}

	/*
	* Set the size budget of the solution set
	* (not used currently)
	*/
	template <class OT>
	void SetSolSizeBudget(SolContainer<OT>& sol, int max_depth, int max_num_nodes) {
		if constexpr (!OT::total_order) {
			return sol->SetSizeBudget(max_depth, max_num_nodes);
		}
	}

	/*
	* Return a copy of the solution
	*/
	template <class OT>
	SolContainer<OT> CopySol
	(const SolContainer<OT>& sol) {
		if constexpr (OT::total_order) {
			return sol;
		} else {
			return std::make_shared < Container<OT> >(*(sol.get()));
		}
	}

	/*
	* Check if the solution is empty
	* single solution: empty feature, empty label
	* solution set: nullptr or zero solutions
	*/
	template <class OT>
	bool CheckEmptySol(const SolContainer<OT>& sol) {
		if constexpr (OT::total_order) {
			return sol.feature == INT32_MAX && sol.label == OT::worst_label;
		} else {
			return sol.get() == nullptr || sol->Size() == 0;
		}
	}

	/*
	* Remove temporary data from a solution before storing it in the cache to safe space and limit copy operations
	*/
	template <class OT>
	inline void SolClearTemp(SolContainer<OT>& sol) {
		if constexpr (!OT::total_order) {
			sol->RemoveTempData();
		}
	}

	/*
	* Add a solution to the solution set (if not dominated) or replace the solution if better
	*/
	template <class OT>
	inline void AddSol(SolContainer<OT>& container, const Node<OT>& sol) {
		if constexpr (OT::total_order) {
			if (sol.solution < container.solution) container = sol;
		} else {
			container->Add(sol);
		}
	}

	/*
	* Add a solution to the solution set (if not dominated) or replace the solution if better
	* If the current node is the root node, use the root-node comparator to determine dominance
	*/
	template <class OT>
	inline void AddSol(OT* task, const int depth, SolContainer<OT>& container, const Node<OT>& sol) {
		if constexpr (OT::total_order) {
			if (sol.solution < container.solution) container = sol;
		} else {
			if (depth == 0) {
				container->AddD0(task, sol);
			} else {
				container->Add(sol);
			}
		}
	}

	/*
	* Add set of solutions to the solution container
	*/
	template <class OT>
	inline void AddSols(SolContainer<OT>& container, const SolContainer<OT>& sols) {
		if constexpr (OT::total_order) {
			if (sols.solution < container.solution) container = sols;
		} else {
			container->Add(*(sols.get()));
		}
	}

	/*
	* Add set of solutions to the solution container
	* If the current node is the root node, use the root-node comparator to determine dominance
	*/
	template <class OT>
	inline void AddSols(OT* task, const int depth, SolContainer<OT>& container, const SolContainer<OT>& sols) {
		if constexpr (OT::total_order) {
			if (sols.solution < container.solution) container = sols;
		} else {
			if (depth == 0) {
				container->AddD0(task, *(sols.get()));
			} else {
				container->Add(*(sols.get()));
			}
		}
	}

	/*
	* Add set of solutions to the solution container
	* Use the inverted dominance operator to determine dominance
	*/
	template <class OT>
	inline void AddSolsInv(SolContainer<OT>& container, const SolContainer<OT>& sols) {
		if constexpr (OT::total_order) {
			if (sols.solution > container.solution) container = sols;
		} else {
			container->AddInv(*(sols.get()));
		}
	}

	/*
	* Add relaxed solutions to the upper bound
	* only if the current node is the root node, and the optimization task has constraints
	* remove info  related to the constraints to increase likelihood of dominance
	*/
	template <class OT>
	inline void AddRootRelaxSols(OT* task, const Branch& branch, SolContainer<OT>& UB, const SolContainer<OT>& container) {
		if constexpr (!OT::total_order && OT::has_constraint) {
			if (branch.Depth() == 0) {
				for (auto& sol : container->GetSolutions()) {
					auto relaxed_sol = sol;
					task->RelaxRootSolution(relaxed_sol);
					AddSol<OT>(UB, relaxed_sol);
				}
			}
		}
	}

	/*
	* Combine two solutions and store the combined solution in out
	*/
	template <class OT>
	inline void CombineSols(int feature, const Node<OT>& left, const Node<OT>& right, const typename OT::SolType& branching_costs, Node<OT>& out) {
		if constexpr (OT::has_branching_costs) {
			out = Node<OT>(feature, OT::Add(branching_costs, OT::Add(left.solution, right.solution)), left.NumNodes(), right.NumNodes());
		} else {
			out = Node<OT>(feature, OT::Add(left.solution, right.solution), left.NumNodes(), right.NumNodes());
		}
	}

	/*
	* Return true iff the left solution value dominates the rigt solution value
	*/
	template <class OT>
	bool LeftDominatesRight(const typename OT::SolType& left, const typename OT::SolType& right) {
		if constexpr (OT::total_order) {
			if constexpr (std::is_same<typename OT::SolType, double>::value) {
				return left * (1 + DBL_DIFF) <= right || std::abs(left - right) <= DBL_DIFF * left;
			}
			return left <= right;
		} else {
			return OT::Dominates(left, right);
		}
	}

	/*
	* Return true iff the left solution value strictly dominates the rigt solution value
	*/
	template <class OT>
	bool LeftStrictDominatesRight(const typename OT::SolType& left, const typename OT::SolType& right) {
		if constexpr (OT::total_order) {
			if constexpr (std::is_same<typename OT::SolType, double>::value) {
				return left * (1 + DBL_DIFF)  < right;
			}
			return left < right;
		} else {
			return OT::Dominates(left, right) && left != right;
		}
	}

	/*
	* Return true iff at least one left solution dominates the right solution
	*/
	template <class OT>
	inline bool LeftDominatesRightSol(const SolContainer<OT>& left, const Node<OT>& right) {
		if constexpr (OT::total_order) {
			return LeftDominatesRight<OT>(left.solution, right.solution);
		} else {
			return left->Dominates(right);
		}
	}

	/*
	* Return true iff at least one left solution strictly dominates the right solution
	*/
	template <class OT>
	inline bool LeftStrictDominatesRightSol(const SolContainer<OT>& left, const Node<OT>& right) {
		if constexpr (OT::total_order) {
			return LeftStrictDominatesRight<OT>(left.solution, right.solution);
		} else {
			return left->StrictDominates(right);
		}
	}

	/*
	* Return true iff for all left solutions there is at least one right solution that inverse dominates it
	*/
	template <class OT>
	inline bool LeftDominatesRight(const SolContainer<OT>& left, const SolContainer<OT>& right) {
		if constexpr (OT::total_order) {
			return LeftDominatesRight<OT>(left.solution, right.solution);
		} else {
			for (const Node<OT>& assignment : left->GetSolutions()) {
				if (!right->DominatesInv(assignment)) { return false; }
			}
			return true;
		}
	}

	/*
	* Return true iff for all left solutions there is at least one right solution that inverse strictly dominates it
	*/
	template <class OT>
	bool LeftStrictDominatesRight(const SolContainer<OT>& left, const SolContainer<OT>& right) {
		if constexpr (OT::total_order) {
			return LeftStrictDominatesRight<OT>(left.solution, right.solution);
		} else {
			for (const auto& assignment : left->GetSolutions()) {
				if (!right->StrictDominatesInv(assignment)) { return false; }
			}
			return true;
		}
	}

	/*
	* Return true if and only if the left and right solution (set) are equal
	*/
	template <class OT>
	bool SolutionsEqual(const SolContainer<OT>& left, const SolContainer<OT>& right) {
		if constexpr (OT::total_order) {
			if constexpr (std::is_same<typename OT::SolType, double>::value) {
				return std::abs(left.solution - right.solution) <= DBL_DIFF * left.solution;
			}
			return left.solution == right.solution;
		} else {
			if (left->Size() == 0 || right->Size() == 0 || left->Size() != right->Size()) return false;
			for (size_t i = 0; i < left->Size(); i++) {
				if (left->GetSolutions()[i].solution != right->GetSolutions()[i].solution) return false;
			}
			return true;
		}
	}

	/*
	* return true iff if the left and right solution and the branching costs equal the solution value of tree->parent
	* If so, store the solution for left and right nodes in tree
	*/
	template<class OT>
	bool CheckReconstructSolution(const Node<OT>& left, const Node<OT>& right, const typename OT::SolType& branching_costs, TreeNode<OT>* tree) {
		static_assert(!(OT::total_order));
		auto combi_sol = OT::Add(left.solution, right.solution);
		if constexpr (OT::has_branching_costs) {
			combi_sol = OT::Add(combi_sol, branching_costs);
		}
		if constexpr (std::is_same<typename OT::SolType, double>::value) {
			if (std::abs(tree->parent.solution - combi_sol) > DBL_DIFF * combi_sol) return false;
		} else {
			if (!(tree->parent.solution == combi_sol)) return false;
		}
		tree->left_child = left;
		tree->right_child = right;
		return true;
	}

	template<class OT>
	void AddValue(SolContainer<OT>& solutions, const typename OT::SolType& value) {
		if constexpr (OT::total_order) {
			OT::Add(solutions.solution, value, solutions.solution);
		} else {
			for (int i = 0; i < solutions.Size(); i++) {
				auto& sol = solutions->GetMutable(i);
				OT::Add(sol.solution, value, sol.solution);
			}
		}
	}

}