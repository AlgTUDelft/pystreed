/**
Partly from Emir Demirovic "MurTree bi-objective"
https://bitbucket.org/EmirD/murtree-bi-objective
*/
#include "solver/cost_storage.h"
#include "solver/solver.h"

namespace STreeD {
	
	template <class OT>
	CostStorage<OT>::CostStorage(int num_features) : num_features(num_features) {
		data2d = std::vector<typename CostStorage<OT>::SolD2Type>(NumElements());// Todo: default value. Store in object also, with default constructor
		//total_costs = implicitly created through default constructor. (todo)
	}

	template <class OT>
	const typename CostStorage<OT>::SolD2Type& CostStorage<OT>::GetCosts(int index_row, int index_column) const {
		runtime_assert(index_row <= index_column);
		int index = IndexSymmetricMatrix(index_row, index_column);
		return data2d[index];
	}

	template <class OT>
	void CostStorage<OT>::ResetToZeros() {
		std::fill(data2d.begin(), data2d.end(), CostStorage<OT>::SolD2Type());
		total_costs = CostStorage<OT>::SolD2Type();
	}

	template <class OT>
	void CostStorage<OT>::ResetToZerosReconstruct(int feature) {
		for (int j = 0; j < num_features; j++) {
			int f1 = feature;
			int f2 = j;
			if (f1 > f2) std::swap(f1, f2);
			data2d[IndexSymmetricMatrix(f1, f2)] = CostStorage<OT>::SolD2Type(); // Must have a default constructor
			data2d[IndexSymmetricMatrix(j, j)] = CostStorage<OT>::SolD2Type();
		}
		total_costs = CostStorage<OT>::SolD2Type();
	}

	template <class OT>
	bool CostStorage<OT>::operator==(const CostStorage<OT>& reference)  const {
		if (num_features != reference.num_features) { return false; }
		if (total_costs != reference.total_costs) { return false; }
		for (int i = 0; i < NumElements(); i++) {
			if (data2d[i] != reference.data2d[i]) { return false; }
		}
		return true;
	}

	template <class OT>
	int CostStorage<OT>::NumElements() const {
		return num_features * (num_features + 1) / 2; //recall that the matrix is symmetric
	}

	template <class OT>
	int CostStorage<OT>::IndexSymmetricMatrix(int index_row, int index_column)  const {
		runtime_assert(index_row <= index_column);
		return num_features * index_row + index_column - index_row * (index_row + 1) / 2;
	}

	template <class OT>
	int CostStorage<OT>::IndexSymmetricMatrixOneDim(int index_row) const {
		return num_features * index_row - index_row * (index_row + 1) / 2;
	}

	template class CostStorage<Accuracy>;
	template class CostStorage<CostComplexAccuracy>;
	template class CostStorage<BalancedAccuracy>;

	template class CostStorage<Regression>;
	template class CostStorage<CostComplexRegression>;
	template class CostStorage<SimpleLinearRegression>;

	template class CostStorage<CostSensitive>;
	template class CostStorage<InstanceCostSensitive>;
	template class CostStorage<F1Score>;
	template class CostStorage<GroupFairness>;
	template class CostStorage<EqOpp>;
	template class CostStorage<PrescriptivePolicy>;
	template class CostStorage<SurvivalAnalysis>;

}