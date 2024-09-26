#pragma once
#include "base.h"

#include "utils/parameter_handler.h"
#include "model/feature_vector.h"
#include "model/data.h"

namespace STreeD {

	class FileReader {
	public:
		
		template <class OT>
		static void ReadData(ParameterHandler& params, AData& data, ADataView& train_data, ADataView& test_data, std::default_random_engine* rng);

		template <class OT>
		static void FillDataView(AData& data, ADataView& view, int start_id, int end_id);
				
		//the DL file format has a instance in each row, where the label at the first position followed by the values of the binary features separated by whitespace
		// for algorithm_selection this is followed by a second line containing runtimes for all possible labels
		//the method reads a file in the DL file format and a BinaryData object
		// duplicate_instances_factor states how many times each instance should be duplicated: this is only useful for testing/benchmarking purposes
		/*
		* the file format has an instance in each row, where the label at the first position followed extra data required
		* by the optimization task, and finishing with the values of the binary features. columsn are separated by whitespaces
		* duplicate_instances_factor states how many times each instance should be duplicated: this is only useful for benchmarking runtime purposes
		*/
		template<class LT, class ET>
		static void ReadFromFile(AData& data, std::string filename, int num_extra_cols,
			int num_instances, int max_num_features, int initial_id, int duplicate_instances_factor);

	};
}