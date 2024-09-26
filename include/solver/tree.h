/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#pragma once
#include "base.h"
#include "model/node.h"
#include "solver/data_splitter.h"

namespace STreeD {
	
	template <class OT>
	struct InternalTrainScore;

	template <class OT>
	struct InternalTestScore;

	template <class OT>
	struct Context;

	class ADataView;

	template <class LT, class ET>
	class Instance;

	template <class OT>
	struct Tree : public std::enable_shared_from_this<Tree<OT>> {
		
		static std::shared_ptr<Tree<OT>> CreateLabelNode(const typename OT::SolLabelType& label) {
			runtime_assert(label != OT::worst_label);
			return std::make_shared<Tree<OT>>(INT32_MAX, label);
		}

		static std::shared_ptr<Tree<OT>> CreateFeatureNodeWithNullChildren(int feature) {
			runtime_assert(feature != INT32_MAX);
			return std::make_shared<Tree<OT>>(feature, OT::worst_label);
		}

		static std::shared_ptr<Tree<OT>> CreateD2TreeFromTreeNodes(const TreeNode<OT>& parent_node, const TreeNode<OT>& left_node, const TreeNode<OT>& right_node) {
			if (parent_node.parent.feature == INT32_MAX) // label node
				return Tree<OT>::CreateLabelNode(parent_node.parent.label);
			auto tree = Tree<OT>::CreateFeatureNodeWithNullChildren(parent_node.parent.feature);
			if (left_node.HasLeftChild() && left_node.HasRightChild()) {
				tree->left_child = Tree<OT>::CreateFeatureNodeWithNullChildren(left_node.parent.feature);
				tree->left_child->left_child = Tree<OT>::CreateLabelNode(left_node.left_child.label);
				tree->left_child->right_child = Tree<OT>::CreateLabelNode(left_node.right_child.label);
			} else {
				tree->left_child = Tree::CreateLabelNode(left_node.parent.label);
			}
			if (right_node.HasLeftChild() && right_node.HasRightChild()) {
				tree->right_child = Tree<OT>::CreateFeatureNodeWithNullChildren(right_node.parent.feature);
				tree->right_child->left_child = Tree<OT>::CreateLabelNode(right_node.left_child.label);
				tree->right_child->right_child = Tree<OT>::CreateLabelNode(right_node.right_child.label);
			} else {
				tree->right_child = Tree<OT>::CreateLabelNode(right_node.parent.label);
			}
			return tree;
		}

		Tree() = delete;
		Tree(int feature, const typename OT::SolLabelType& label) : feature(feature), label(label), left_child(nullptr), right_child(nullptr) {}

		int Depth() const {
			if (IsLabelNode()) { return 0; }
			return 1 + std::max(left_child->Depth(), right_child->Depth());
		}

		int NumNodes() const {
			if (IsLabelNode()) { return 0; }
			return 1 + left_child->NumNodes() + right_child->NumNodes();
		}

		bool IsLabelNode() const { return label != OT::worst_label; }
		bool IsFeatureNode() const { return feature != INT32_MAX; }

		void ComputeTrainScore(DataSplitter* data_splitter, OT* task, const typename OT::ContextType& context,
			const ADataView& train_data, InternalTrainScore<OT>& result) const {
			result.average_path_length += train_data.Size();
			if (IsLabelNode()) {
				result.train_value = OT::Add(result.train_value, task->GetLeafCosts(train_data, context, label));
				result.train_test_value = OT::TestAdd(result.train_test_value, task->GetTestLeafCosts(train_data, context, label));
				return;
			}
			typename OT::ContextType left_context, right_context;
			task->GetLeftContext(train_data, context, feature, left_context); 
			task->GetRightContext(train_data, context, feature, right_context);
			ADataView left_train_data, right_train_data;
			data_splitter->Split(train_data, context.GetBranch(), feature, left_train_data, right_train_data);
			if constexpr (OT::has_branching_costs) {
				result.train_value = OT::Add(result.train_value, task->GetBranchingCosts(train_data, context, feature));
				result.train_test_value = OT::TestAdd(result.train_test_value, task->GetTestBranchingCosts(train_data, context, feature));
			}
			left_child->ComputeTrainScore(data_splitter, task, left_context, left_train_data, result);
			right_child->ComputeTrainScore(data_splitter, task, right_context, right_train_data, result);
		}

		void ComputeTestScore(DataSplitter* data_splitter, OT* task, const typename OT::ContextType& context,
			const std::vector<int>& flipped_features, const ADataView& test_data, InternalTestScore<OT>& result) const {
			result.average_path_length += test_data.Size();
			if (IsLabelNode()) {
				result.test_test_value = OT::TestAdd(result.test_test_value, task->GetTestLeafCosts(test_data, context, label));
				return;
			}
			typename OT::ContextType left_context, right_context;
			task->GetLeftContext(test_data, context, feature, left_context);
			task->GetRightContext(test_data, context, feature, right_context);
			ADataView left_test_data, right_test_data;
			data_splitter->Split(test_data, context.GetBranch(), feature, left_test_data, right_test_data, true);
			if constexpr (OT::has_branching_costs) {
				result.test_test_value = OT::TestAdd(result.test_test_value, task->GetTestBranchingCosts(test_data, context, feature));
			}
			if (flipped_features.size() > feature && flipped_features[feature] == 1) {
				right_child->ComputeTestScore(data_splitter, task, left_context, flipped_features, left_test_data, result);
				left_child->ComputeTestScore(data_splitter, task, right_context, flipped_features, right_test_data, result);
			} else {
				left_child->ComputeTestScore(data_splitter, task, left_context, flipped_features, left_test_data, result);
				right_child->ComputeTestScore(data_splitter, task, right_context, flipped_features, right_test_data, result);
			}
		}

		void Classify(DataSplitter* data_splitter, OT* task, const typename OT::ContextType& context, const std::vector<int>& flipped_features, const ADataView& data, std::vector<typename OT::LabelType>& labels) {
			if (IsLabelNode()) {
				for (int k = 0; k < data.NumLabels(); k++) {
					for (auto i : data.GetInstancesForLabel(k)) {
						labels[i->GetID()] = task->Classify(i, label);
					}
				}
			} else {
				typename OT::ContextType left_context, right_context;
				task->GetLeftContext(data, context, feature, left_context);
				task->GetRightContext(data, context, feature, right_context);
				ADataView left_data, right_data;
				data_splitter->Split(data, context.GetBranch(), feature, left_data, right_data, true);
				if (flipped_features[feature] == 1) {
					right_child->Classify(data_splitter, task, left_context, flipped_features, left_data, labels);
					left_child->Classify(data_splitter, task, right_context, flipped_features, right_data, labels);
				} else {
					left_child->Classify(data_splitter, task, left_context, flipped_features, left_data, labels);
					right_child->Classify(data_splitter, task, right_context, flipped_features, right_data, labels);
				}
			}
		}

		std::string ToString() const {
			std::stringstream ss;
			BuildTreeString(ss);
			return ss.str();
		}

		void BuildTreeString(std::stringstream& ss) const {
			if (IsLabelNode()) {
				ss << "[" << OT::LabelToString(label) << "]";
			} else {
				ss << "[" << feature << ",";
				left_child->BuildTreeString(ss);
				ss << ",";
				right_child->BuildTreeString(ss);
				ss << "]";
			}
		}

		void FlipFlippedFeatures(const std::vector<int>& flipped_features) {
			if (feature >= flipped_features.size()) return; // In a leaf node, or when hypertuning
			if (flipped_features[feature]) std::swap(left_child, right_child);
			left_child->FlipFlippedFeatures(flipped_features);
			right_child->FlipFlippedFeatures(flipped_features);
		}

		int feature{ INT32_MAX };
		typename OT::SolLabelType label{ OT::worst_label };
		std::shared_ptr<Tree<OT>> left_child, right_child;
	};

}