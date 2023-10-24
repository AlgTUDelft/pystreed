/**
Partly from Emir Demirovic "MurTree bi-objective"
https://bitbucket.org/EmirD/murtree-bi-objective
*/
#pragma once
#include "base.h"
#include "model/data.h"

namespace STreeD {
	
	template <class OT>
	class CostStorage {
	public:
		using SolD2Type = typename OT::SolD2Type;

		CostStorage() : num_features(0), total_costs(SolD2Type()) {}
		CostStorage(int num_features);

		inline const SolD2Type& GetTotalCosts() const { return total_costs; }
		const SolD2Type& GetCosts(int index_row, int index_column) const;
		inline const SolD2Type& GetCosts(int index) const { return data2d[index]; }
		int IndexSymmetricMatrix(int index_row, int index_column) const;
		int IndexSymmetricMatrixOneDim(int index_row) const;

		inline void UpdateTotalCosts(const SolD2Type& values) { total_costs += values; }
		inline void UpdateCosts(int index_row, int index_column, const SolD2Type& val) { data2d[IndexSymmetricMatrix(index_row, index_column)] += val; }
		inline void UpdateCosts(int index, const SolD2Type& val) { data2d[index] += val; }

		void ResetToZeros();
		void ResetToZerosReconstruct(int feature);

		bool operator==(const CostStorage& reference)  const;

	private:
		int NumElements() const;

		std::vector<SolD2Type> data2d;
		SolD2Type total_costs;
		int num_features{ 0 };
	};
}