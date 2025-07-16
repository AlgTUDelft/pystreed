/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#pragma once

#include "base.h"
#include "solver/cache.h"
#include "solver/difference_computer.h"
#include "model/data.h"
#include "model/branch.h"

namespace STreeD {
	template <class OT>
	struct PairLowerBoundOptimal {
		using SolContainer = typename std::conditional<OT::total_order, Node<OT>, std::shared_ptr<Container<OT>>>::type;

		PairLowerBoundOptimal(SolContainer& lb, bool opt) :lower_bound(lb), optimal(opt) {}
		SolContainer lower_bound;
		bool optimal;
	};

	template <class OT>
	struct PairWorstCount {
		using SolType = typename OT::SolType;

		PairWorstCount(SolType& sub, int diff) :subtract(sub), total_difference(diff) {}
		SolType subtract;
		int total_difference;
	};

	template <class OT>
	class SimilarityLowerBoundComputer {
		using SolType = typename OT::SolType;
		using SolContainer = typename std::conditional<OT::total_order, Node<OT>, std::shared_ptr<Container<OT>>>::type;

	public:
		SimilarityLowerBoundComputer(OT* optimization_task, int num_labels, int max_depth, int num_nodes, int num_instances);

		//computes the lower bound with respect to the data currently present in the data structure
		//note that this does not add any information internally
		//use 'UpdateArchive' to add datasets to the data structure
		PairLowerBoundOptimal<OT> ComputeLowerBound(ADataView& data, const Branch& branch, int depth, int num_nodes, Cache<OT>* cache);

		//adds the data, possibly replacing some previously stored dataset in case there the data structure is full. 
		//when replacing, it will find the most similar dataset and replace it with the input
		//TODO make it should replace the most disimilar dataset, and not the most similar?
		void UpdateArchive(ADataView& data, const Branch& branch, int depth);

		void Disable();

		void Reset();

	private:

		struct ArchiveEntry {
			ArchiveEntry(ADataView& d, const Branch& b) :
				data(d),
				branch(b) {
			}

			ADataView data;
			Branch branch;
		};

		void Initialise(OT* optimization_task, int num_labels, int max_depth, int num_nodes);
		ArchiveEntry& GetMostSimilarStoredData(ADataView& data, int depth);
		SolContainer SubstractLB(SolContainer& lb, SolType& values) const;

		std::vector<std::vector<ArchiveEntry> > archive_;//archive_[depth][i]
		bool disabled_;

		std::vector<SolType> _worst;
		bool _best_zero;

		OT* task{ nullptr };
	};
}