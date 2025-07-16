/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#include "utils/file_reader.h"
#include "tasks/tasks.h"

namespace STreeD {
	
	template <class OT>
	void CopyTrainData(AData& data, ADataView& train_data, ADataView& test_data) {
		std::vector<std::vector<const AInstance*>> instances_per_label;
		instances_per_label.resize(train_data.NumLabels());
		int id = data.Size();
		for (int k = 0; k < train_data.NumLabels(); k++) {
			for (auto& instance : train_data.GetInstancesForLabel(k)) {
				auto inst = static_cast<const Instance<typename OT::LabelType, typename OT::ET>*>(instance);
				auto new_instance = new Instance<typename OT::LabelType, typename OT::ET>(*inst);
				new_instance->SetID(id++);
				instances_per_label[k].push_back(new_instance);
				data.AddInstance(new_instance);
			}
		}
		test_data = ADataView(&data, instances_per_label, {});
	}

	// Read the file(s) specified in parameters, stores the instances in data, and provides train and test dataviews in train_data and test_data
	template <class OT>
	void FileReader::ReadData(ParameterHandler& parameters, AData& data, ADataView& train_data, ADataView& test_data, std::default_random_engine* rng) {
		std::string train_file = parameters.GetStringParameter("file");
		std::string test_file = parameters.GetStringParameter("test-file");
		int num_extra_cols = int(parameters.GetIntegerParameter("num-extra-cols"));
		int num_instances = int(parameters.GetIntegerParameter("num-instances"));
		int max_num_features = int(parameters.GetIntegerParameter("max-num-features"));
		int duplicate_factor = int(parameters.GetIntegerParameter("duplicate-factor"));
		double train_test_split = parameters.GetFloatParameter("train-test-split");
		bool stratify = parameters.GetBooleanParameter("stratify");

		FileReader::ReadFromFile<typename OT::LabelType, typename OT::ET>(data, train_file, num_extra_cols, num_instances, max_num_features, 0, duplicate_factor);
		int train_size = data.Size();

		if (test_file != "") {
			FileReader::ReadFromFile<typename OT::LabelType, typename OT::ET>(data, test_file, num_extra_cols, INT32_MAX, max_num_features, train_size, 1);
			FillDataView<OT>(data, train_data, 0, train_size);
			FillDataView<OT>(data, test_data, train_data.Size(), data.Size());
		} else {
			FillDataView<OT>(data, train_data, 0, train_size);
			if (train_test_split <= DBL_EPSILON) {
				CopyTrainData<OT>(data, train_data, test_data);
			} else {
				ADataView all_data = train_data;
				ADataView::TrainTestSplitData<typename OT::LabelType>(all_data, train_data, test_data, rng, train_test_split, stratify);
			}
		}
	}

	template <class OT>
	void FileReader::FillDataView(AData& data, ADataView& view, int start_id, int end_id) {
		const bool regression = std::is_same<typename OT::LabelType, double>::value;
		std::vector<std::vector<const AInstance*>> instances_per_label;
		if (regression) instances_per_label.resize(1);

		for (AInstance* instance : data.GetInstances()) {
			if (instance->GetID() < start_id) continue;
			if (instance->GetID() >= end_id) continue;

			if (regression) {
				instances_per_label[0].push_back(instance);
			} else {
				int i_label = GetInstanceLabel<int>(instance);
				if (instances_per_label.size() <= i_label) { instances_per_label.resize(i_label + 1); }
				instances_per_label[i_label].push_back(instance);
			}
		}

		view = ADataView(&data, instances_per_label, {});
	}

	template<class LT, class ET>
	void FileReader::ReadFromFile(AData& data, std::string filename, int num_extra_cols,
		int num_instances, int max_num_features, int initial_id, int duplicate_instances_factor) {
		
		std::ifstream file(filename.c_str());

		if (!file) { std::cout << "Error: File " << filename << " does not exist!\n"; runtime_assert(file); }

		std::string line;
		int id = initial_id;
		int num_features = INT32_MAX;
		bool include_all = num_instances == INT32_MAX;
		int available_instances = INT32_MAX;

		// Count lines and select random instances, if !include_all
		std::vector<int> indices;
		if (!include_all) {
			available_instances = 0;
			while (std::getline(file, line)) {
				available_instances++;
			}
			file.clear();
			file.seekg(0);
			if (available_instances < num_instances) {
				include_all = true;
			} else {
				indices.resize(available_instances);
				std::iota(indices.begin(), indices.end(), 0);
				std::shuffle(indices.begin(), indices.end(), std::random_device()); // TODO use the same random device
				indices.resize(num_instances);
				std::sort(indices.begin(), indices.end());
			}
		}

		std::vector<const AInstance*> instances;
		int line_no = -1;
		int indices_no = 0;
		while (std::getline(file, line)) {
			line_no++;			
			if (!include_all && line_no != indices[indices_no]) continue;
			indices_no++;

			std::istringstream iss(line);
			//the first value in the line is the label,
			// followed by 0-1 features
			LT label;
			iss >> label;

			// Read extra instance data
			auto extra_data = ET::ReadData(iss, num_extra_cols);

			std::getline(iss, line);
			runtime_assert(num_features == INT32_MAX || num_features == int((line.size()) / 2) || num_features == max_num_features);
			if (num_features == INT32_MAX) { num_features = int((line.size()) / 2); }
			num_features = std::min(num_features, max_num_features);
			iss = std::istringstream(line);

			std::vector<bool> v(num_features);
			for (int i = 0; i < num_features; i++) {
				uint32_t temp;
				iss >> temp;
				if (temp != 0 && temp != 1) {
					std::cout << "Error: Encountered unexpected non-binary symbol '" << temp << "' at line " << line_no << " in " << filename << "." << std::endl;
					exit(1);
				}
				if (i >= data.NumFeatures()) continue;
				v[i] = temp;
			}

			for (int i = 0; i < duplicate_instances_factor; i++) {
				auto instance = new Instance<LT, ET>(id, v, label, extra_data);
				data.AddInstance(instance);
				id++;
				if (id - initial_id >= num_instances) break;
			}
			if (id - initial_id >= num_instances) break;
		}

		if (data.NumFeatures() > num_features) data.SetNumFeatures(num_features);
	}


	template void FileReader::ReadData<Accuracy>(ParameterHandler&, AData&, ADataView&, ADataView&, std::default_random_engine*);
	template void FileReader::ReadData<CostComplexAccuracy>(ParameterHandler&, AData&, ADataView&, ADataView&, std::default_random_engine*);
	template void FileReader::ReadData<BalancedAccuracy>(ParameterHandler&, AData&, ADataView&, ADataView&, std::default_random_engine*);

	template void FileReader::ReadData<Regression>(ParameterHandler&, AData&, ADataView&, ADataView&, std::default_random_engine*);
	template void FileReader::ReadData<CostComplexRegression>(ParameterHandler&, AData&, ADataView&, ADataView&, std::default_random_engine*);
	template void FileReader::ReadData<PieceWiseLinearRegression>(ParameterHandler&, AData&, ADataView&, ADataView&, std::default_random_engine*);
	template void FileReader::ReadData<SimpleLinearRegression>(ParameterHandler&, AData&, ADataView&, ADataView&, std::default_random_engine*);

	template void FileReader::ReadData<CostSensitive>(ParameterHandler&, AData&, ADataView&, ADataView&, std::default_random_engine*);
	template void FileReader::ReadData<InstanceCostSensitive>(ParameterHandler&, AData&, ADataView&, ADataView&, std::default_random_engine*);
	template void FileReader::ReadData<F1Score>(ParameterHandler&, AData&, ADataView&, ADataView&, std::default_random_engine*);
	template void FileReader::ReadData<GroupFairness>(ParameterHandler&, AData&, ADataView&, ADataView&, std::default_random_engine*);
	template void FileReader::ReadData<EqOpp>(ParameterHandler&, AData&, ADataView&, ADataView&, std::default_random_engine*);
	template void FileReader::ReadData<PrescriptivePolicy>(ParameterHandler&, AData&, ADataView&, ADataView&, std::default_random_engine*);
	template void FileReader::ReadData<SurvivalAnalysis>(ParameterHandler&, AData&, ADataView&, ADataView&, std::default_random_engine*);

}