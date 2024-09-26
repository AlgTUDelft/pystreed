/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#pragma once
#include "base.h"
#include "model/container.h"
#include "solver/cost_storage.h"
#include "solver/difference_computer.h"
#include "model/branch.h"
#include "solver/counter.h"
#include "solver/optimization_utils.h"
#include "tasks/tasks.h"

namespace STreeD {
		
	template<class OT>
	struct Sols {
		using SolType = typename OT::SolType;
		SolType sol00, sol01, sol10, sol11;
	};

	struct IndexInfo {
		int ix_f1f1{ 0 };
		int ix_f1f2{ 0 };
		int ix_f2f2{ 0 };
		bool swap{ false };
		bool equal{ false };
	};

	template <class OT>
	class CostCalculator {
	public:
		using SolType = typename OT::SolType;
		using SolD2Type = typename OT::SolD2Type;
		using BranchSolD2Type = typename OT::BranchSolD2Type;
		using LabelType = typename OT::LabelType;
		using SolLabelType = typename OT::SolLabelType;
		using Context = typename OT::ContextType;

		CostCalculator() = delete;
		CostCalculator(OT* task, int num_features, int num_labels, const std::vector<int>& feature_order);

		bool Initialize(const ADataView& data, const Context& branch, int num_nodes);
		void InitializeReconstruct(const ADataView& data, const Context& branch, int feature);
		int ProbeDifference(const ADataView& data) const;
		void InitializeIndexInfos(int num_features);

		void GetIndexInfo(int f1, int f2, IndexInfo& index_info) const { index_info = index_infos[f1][f2]; }
		void CalcLeafSol(SolType& sol, int label, SolLabelType& label_out) const;
		void CalcSols(const Counts& counts, Sols<OT>& sols, int label, int f1, int f2) const;
		void CalcSols(const Counts& counts, Sols<OT>& sols, int label, const IndexInfo& index) const;
		void CalcSol00(SolType& sol, int label, int f1, int f2) const;
		void CalcSol11(SolType& sol, int label, int f1, int f2) const;

		const SolD2Type GetCosts00(int label, int f1, int f2) const;
		const SolD2Type GetCosts01(int label, int f1, int f2) const;
		const SolD2Type GetCosts10(int label, int f1, int f2) const;
		const SolD2Type GetCosts11(int label, int f1, int f2) const;

		const SolLabelType GetLabel00(int label, int f1, int f2) const;
		const SolLabelType GetLabel01(int label, int f1, int f2) const;
		const SolLabelType GetLabel10(int label, int f1, int f2) const;
		const SolLabelType GetLabel11(int label, int f1, int f2) const;
		const SolLabelType GetLabel(int label, const SolD2Type& costs, int count) const;

		const SolType GetBranchingCosts(int f1) const;
		const SolType GetBranchingCosts0(int count0, int f1, int f2) const;
		const SolType GetBranchingCosts1(int count1, int f1, int f2) const;
		void ResetBranchingCosts();

		int GetTotalCount() const { return counter.GetTotalCount(); }
		void GetCounts(Counts& counts, int f1, int f2) const;
		void GetCounts(Counts& counts, const IndexInfo& index) const;
		int GetCount00(int f1, int f2) const;
		int GetCount01(int f1, int f2) const;
		int GetCount10(int f1, int f2) const;
		int GetCount11(int f1, int f2) const;

		void UpdateCosts(const ADataView& data, int value);
		void UpdateCostsReconstruct(const ADataView& data, int feature);
		void UpdateBranchingCosts(const ADataView& data, const Context& context);

	private:
		OT* task;
		ADataView data;
		int num_nodes, num_features;
		std::vector<CostStorage<OT>> cost_storage;
		std::vector<std::vector<BranchSolD2Type>> branching_costs;
		Counter counter;
		std::vector<std::vector<IndexInfo>> index_infos;
		mutable typename OT::SolD2Type temp_costs1, temp_costs2;
		mutable std::vector<int> labels;
		mutable ADataView data_add, data_remove;
		std::vector<int> feature_order;
	};

}