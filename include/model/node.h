#pragma once
#include "base.h"

namespace STreeD {

	// A node represents a solution to a subproblem.
	template<class OT>
	struct Node {

		using SolType = typename OT::SolType;
		using SolLabelType = typename OT::SolLabelType;

		// Initialize the node with all attributes initialized to their default values
		Node() : feature(INT32_MAX), label(OT::worst_label), solution(OT::worst), num_nodes_left(INT32_MAX), num_nodes_right(INT32_MAX) {}

		// Initialize a leaf node with the OT::worst_label solution, and the given solution value
		Node(const SolType& solution) : feature(INT32_MAX), label(OT::worst_label), solution(solution), num_nodes_left(INT32_MAX), num_nodes_right(INT32_MAX) {}

		// Initialize a leaf node with the given label and solution value
		Node(const SolLabelType& label, const SolType& solution)
			: feature(INT32_MAX), label(label), solution(solution), num_nodes_left(0), num_nodes_right(0) {}

		// Initialize a branching node with the given feature, solution value, and left and right children
		Node(int feature, const SolType& solution, int num_nodes_left, int num_nodes_right)
			: feature(feature), label(OT::worst_label), solution(solution), num_nodes_left(num_nodes_left), num_nodes_right(num_nodes_right) {
		}

		// Set the values of this node
		void Set(int feature, const SolLabelType& label, const SolType& solution, int num_nodes_left, int num_nodes_right) {
			this->feature = feature; this->solution = solution; this->label = label; this->num_nodes_left = num_nodes_left; this->num_nodes_right = num_nodes_right;
		}

		// Return the number of branching nodes in the tree with this node as its root (including this node)
		inline int NumNodes() const { return feature == INT32_MAX ? 0 : num_nodes_left + num_nodes_right + 1; }
		
		// Return True if this node is infeasible (solution == OT::worst)
		inline bool IsInfeasible() const { return !IsFeasible(); }

		// Return True if this node is feasible (solution != OT::worst)
		inline bool IsFeasible() const { return solution != OT::worst; }

		// Return true if this node is equal to another node in all its attributes
		inline bool operator==(const Node<OT>& other) const {
			return feature == other.feature && label == other.label && solution == other.solution
				&& num_nodes_left == other.num_nodes_left && num_nodes_right == other.num_nodes_right;
		}

		// The feature to branch on (INT32_MAX if the node is a leaf node)
		int feature;
		// The label of the leaf node or (OT::worst_label) if the node is branching node.
		SolLabelType label;
		// The solution value of the node (default = OT::worst, means infeasible)
		SolType solution;

		// The number of left children of this node
		int num_nodes_left;

		// The number of right childre nof this node
		int num_nodes_right;
	};

	// A tree node stores a root node, and a left and right child node.
	// Used for reconstructing the tree
	template<class OT>
	struct TreeNode {
		TreeNode() = default;
		TreeNode(const Node<OT>& parent) : parent(parent) {}
		TreeNode(const Node<OT>& parent, const Node<OT>& left, const Node<OT>& right)
			: parent(parent), left_child(left), right_child(right) {}
		
		void Set(const Node<OT>& parent, const Node<OT>& left, const Node<OT>& right) {
			this->parent = parent; this->left_child = left; this->right_child = right;
		}

		inline bool HasLeftChild() const { return parent.feature != INT32_MAX; }
		inline bool HasRightChild() const { return parent.feature != INT32_MAX; }


		Node<OT> parent, left_child, right_child;
	};
}