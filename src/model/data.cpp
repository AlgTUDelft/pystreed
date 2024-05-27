#include "model/data.h"

namespace STreeD {

	ADataView::ADataView(const AData* data, const std::vector<std::vector<const AInstance*>>& instances, const std::vector<std::vector<double>>& instance_weights)
		: data(data), instances(instances), instance_weights(instance_weights)
	{
		size = 0;
		for (const auto& v : instances) size += int(v.size());
	}

	void ADataView::SplitData(int feature, ADataView& left, ADataView& right) const {
		left.data = data;
		right.data = data;
		left.instances.resize(NumLabels());
		right.instances.resize(NumLabels());
		left.size = 0;
		right.size = 0;
		//std::vector<std::vector<std::vector<int>>> label_indices(NumLabels(), std::vector<std::vector<int>>(2)); // second index: 0=left, 1=right
		for (int label = 0; label < NumLabels(); label++) {
			auto& _instances = GetInstancesForLabel(label);
			for (auto& instance : _instances) {
				if (instance->IsFeaturePresent(feature)) {
					right.instances[label].push_back(instance);
				} else {
					left.instances[label].push_back(instance);
				}
			}
			left.size += int(left.instances[label].size());
			right.size += int(right.instances[label].size());
			runtime_assert(left.instances[label].size() + right.instances[label].size() == _instances.size());
		}
		runtime_assert(left.size + right.size == size);
	}

	template<class LT>
	void ADataView::TrainTestSplitData(const ADataView& data, ADataView& train, ADataView& test, std::default_random_engine* rng, double test_percentage, bool stratify) {
		runtime_assert(test_percentage >= 0.0 && test_percentage <= 1.0);
		const int num_labels = data.NumLabels();
		const bool regression = std::is_same<LT, double>::value;
		
		train.Clear();
		test.Clear();
		train.data = data.GetData();
		test.data = data.GetData();
		train.instances.resize(num_labels); // todo weights
		test.instances.resize(num_labels);  // todo weights
		std::vector<std::vector<int>> train_label_indices(num_labels);
		std::vector<std::vector<int>> test_label_indices(num_labels);

		if (stratify) { // Apply Stratification
			
			if (regression) {
				int _num_labels = 10;
				std::vector<std::vector<int>> all_label_indices(_num_labels);

				auto instances = data.GetInstancesForLabel(0); // Copy vector
				std::sort(instances.begin(), instances.end(), [](const AInstance*& i1, const AInstance*& i2) {
					return GetInstanceLabel<double>(i1) > GetInstanceLabel<double>(i2);
				});
				int i = 0;
				int bin_size = int(instances.size()) / _num_labels;
				for (auto& instance : instances) {
					all_label_indices[std::min(i++ / bin_size, _num_labels-1)].push_back(instance->GetID());
				}

				for (int y = 0; y < _num_labels; y++) {
					std::vector<int> indices;
					auto& label_indices = all_label_indices[y];
					indices.assign(label_indices.begin(), label_indices.end());
					std::shuffle(indices.begin(), indices.end(), *rng);
					const int test_split_index = int(std::round(test_percentage * indices.size()));

					auto begin = indices.begin();
					auto split = indices.begin() + test_split_index;
					auto end = indices.end();
					test_label_indices[0].insert(test_label_indices[0].end(), begin, split);
					train_label_indices[0].insert(train_label_indices[0].end(), split, end);
				}

			} else {

				std::vector<std::vector<int>> all_label_indices(num_labels);

				for (int y = 0; y < num_labels; y++) {
					for (auto& instance : data.GetInstancesForLabel(y)) {
						int label = int(GetInstanceLabel<LT>(instance));
						all_label_indices[label].push_back(instance->GetID());
					}
				}

				for (int y = 0; y < num_labels; y++) {
					std::vector<int> indices;
					auto& label_indices = all_label_indices[y];
					indices.assign(label_indices.begin(), label_indices.end());
					std::shuffle(indices.begin(), indices.end(), *rng);
					const int test_split_index = int(std::round(test_percentage * indices.size()));

					auto begin = indices.begin();
					auto split = indices.begin() + test_split_index;
					auto end = indices.end();
					test_label_indices[y].insert(test_label_indices[y].end(), begin, split);
					train_label_indices[y].insert(train_label_indices[y].end(), split, end);
				}

			}
	
		} else {
			std::vector<int> test_indices, train_indices;
			std::vector<int> indices;
			for (int y = 0; y < num_labels; y++) {
				for (auto& instance : data.GetInstancesForLabel(y)) {
					indices.push_back(instance->GetID());
				}
			}

			std::shuffle(indices.begin(), indices.end(), *rng);
			const int test_split_index = int(std::round(test_percentage * indices.size()));

			auto begin = indices.begin();
			auto split = indices.begin() + test_split_index;
			auto end = indices.end();
			test_indices.assign(begin, split);
			train_indices.assign(split, end);
			if (regression) {
				train_label_indices[0] = train_indices;
				test_label_indices[0] = test_indices;
			} else {
				for (auto i : test_indices) {
					int label = int(GetInstanceLabel<LT>(data.GetData()->GetInstance(i)));
					test_label_indices[label].push_back(i);
				}
				for (auto i : train_indices) {
					int label = int(GetInstanceLabel<LT>(data.GetData()->GetInstance(i)));
					train_label_indices[label].push_back(i);
				}
			}
		}

		train.size = 0;
		test.size = 0;
		for (int j = 0; j < num_labels; j++) {
			std::sort(train_label_indices[j].begin(), train_label_indices[j].end());
			std::sort(test_label_indices[j].begin(), test_label_indices[j].end());
			for (auto i : train_label_indices[j])
				train.instances[j].push_back(data.GetData()->GetInstance(i));
			for (auto i : test_label_indices[j])
				test.instances[j].push_back(data.GetData()->GetInstance(i));

			train.size += int(train.instances[j].size());
			test.size += int(test.instances[j].size());
		}
	}

	template<class LT>
	void ADataView::KFoldSplit(const ADataView& data, std::vector<ADataView>& train_folds, std::vector<ADataView>& test_folds, std::default_random_engine* rng, int folds, bool stratify) {
		const int num_labels = data.NumLabels();
		const double fold_percentage = 1.0 / folds;
		const bool regression = std::is_same<LT, double>::value;

		

		train_folds.clear();
		test_folds.clear();

		std::vector<std::vector<std::vector<int>>> fold_label_indices(folds, std::vector<std::vector<int>>(num_labels));
		
		if (stratify) { // Apply Stratification

			if (regression) {
				int _num_labels = 10;
				std::vector<std::vector<int>> all_label_indices(_num_labels);

				auto instances = data.GetInstancesForLabel(0); // Copy vector
				std::sort(instances.begin(), instances.end(), [](const AInstance*& i1, const AInstance*& i2) {
					return GetInstanceLabel<double>(i1) > GetInstanceLabel<double>(i2);
				});
				int i = 0;
				int bin_size = int(instances.size()) / _num_labels;
				for (auto& instance : instances) {
					all_label_indices[std::min(i++ / bin_size, _num_labels - 1)].push_back(instance->GetID());
				}

				for (int y = 0; y < _num_labels; y++) {
					std::vector<int> indices;
					auto& label_indices = all_label_indices[y];
					indices.assign(label_indices.begin(), label_indices.end());
					std::shuffle(indices.begin(), indices.end(), *rng);
					int begin_index = 0;
					for (int fold = 0; fold < folds; fold++) {
						int end_index = int(std::round(fold_percentage * (fold + 1) * indices.size()));
						auto begin = indices.begin() + begin_index;
						auto end = indices.begin() + end_index;
						if (fold == folds - 1) end = indices.end();
						fold_label_indices[fold][0].insert(fold_label_indices[fold][0].end(), begin, end);
						begin_index = end_index;
					}
				}

			} else {

				std::vector<std::vector<int>> all_label_indices(num_labels);

				for (int y = 0; y < num_labels; y++) {
					for (auto& instance : data.GetInstancesForLabel(y)) {
						int label = int(GetInstanceLabel<LT>(instance));
						all_label_indices[label].push_back(instance->GetID());
					}
				}

				for (int y = 0; y < num_labels; y++) {
					std::vector<int> indices;
					auto& label_indices = all_label_indices[y];
					indices.assign(label_indices.begin(), label_indices.end());
					std::shuffle(indices.begin(), indices.end(), *rng);
					int begin_index = 0;
					for (int fold = 0; fold < folds; fold++) {
						int end_index = int(std::round(fold_percentage *(fold + 1) * indices.size()));
						auto begin = indices.begin() + begin_index;
						auto end = indices.begin() + end_index;
						if (fold == folds - 1) end = indices.end();
						fold_label_indices[fold][0].insert(fold_label_indices[fold][0].end(), begin, end);
						begin_index = end_index;
					}
				}

			}

		} else {
			std::vector<std::vector<int>> fold_indices(folds);
			std::vector<int> indices;
			for (int y = 0; y < num_labels; y++) {
				for (auto& instance : data.GetInstancesForLabel(y)) {
					indices.push_back(instance->GetID());
				}
			}

			std::shuffle(indices.begin(), indices.end(), *rng);
			int begin_index = 0;
			for (int fold = 0; fold < folds; fold++) {
				int end_index = int(std::round(fold_percentage * (fold + 1) * indices.size()));
				auto begin = indices.begin() + begin_index;
				auto end = indices.begin() + end_index;
				if (fold == folds - 1) end = indices.end();
				fold_indices[fold].insert(fold_indices[fold].end(), begin, end);
				begin_index = end_index;
			}

			if (regression) {
				for (int fold = 0; fold < folds; fold++) {
					fold_label_indices[fold][0] = fold_indices[fold];
				}
			} else {
				for (int fold = 0; fold < folds; fold++) {
					for (auto i : fold_indices[fold]) {
						int label = int(GetInstanceLabel<LT>(data.GetData()->GetInstance(i)));
						fold_label_indices[fold][label].push_back(i);
					}
				}
			}
		}



		for (int fold = 0; fold < folds; fold++) {
			train_folds.push_back(ADataView(data.GetData(), num_labels));
			test_folds.push_back(ADataView(data.GetData(), num_labels));

			ADataView& train = train_folds[fold];
			ADataView& test = test_folds[fold];
			train.size = 0;
			test.size = 0;

			std::vector<std::vector<int>> train_label_indices(num_labels), test_label_indices(num_labels);
			for (int fold2 = 0; fold2 < folds; fold2++) {
				if (fold2 == fold) continue;
				for (int j = 0; j < num_labels; j++) {
					train_label_indices[j].insert(train_label_indices[j].end(), fold_label_indices[fold2][j].begin(), fold_label_indices[fold2][j].end());
				}
			}
			for (int j = 0; j < num_labels; j++) {
				test_label_indices[j].insert(test_label_indices[j].end(), fold_label_indices[fold][j].begin(), fold_label_indices[fold][j].end());
				
				std::sort(train_label_indices[j].begin(), train_label_indices[j].end());
				std::sort(test_label_indices[j].begin(), test_label_indices[j].end());
				for (auto i : train_label_indices[j])
					train.instances[j].push_back(data.GetData()->GetInstance(i));
				for (auto i : test_label_indices[j])
					test.instances[j].push_back(data.GetData()->GetInstance(i));

				train.size += int(train.instances[j].size());
				test.size += int(test.instances[j].size());
			}

		}

		
		
	}

	void ADataView::AddInstance(int label, const AInstance* instance) {
		if(bitset_view.IsBitViewSet()) bitset_view = ADataViewBitSet();
		instances[label].push_back(instance);
		size += 1;
	}

	void ADataView::ComputeSize() {
		size = 0;
		for (int label = 0; label < NumLabels(); label++) {
			size += NumInstancesForLabel(label);
		}
	}

	bool ADataView::operator==(const ADataView& other) const {
		if (Size() == 0 || other.Size() == 0) return false; // Not initialized
		runtime_assert(NumLabels() == other.NumLabels() && NumFeatures() == other.NumFeatures());

		if (IsHashSet() && other.IsHashSet() && GetHash() != other.GetHash()) { return false; }

		//basic check on the size
		if (Size() != other.Size()) { return false; }
		//basic check on the size of each individual label
		for (int label = 0; label < NumLabels(); label++) {
			if (NumInstancesForLabel(label) != other.NumInstancesForLabel(label)) return false;
		}

		//now compare individual feature vectors
		//note that the indicies are kept sorted in the data

		for (int label = 0; label < NumLabels(); label++) {
			auto& v1 = GetInstancesForLabel(label);
			auto& v2 = other.GetInstancesForLabel(label);
			for (size_t i = 0; i < v1.size(); i++) {
				if (v1[i]->GetID() != v2[i]->GetID()) return false;
			}
		}

		return true;
	}

	ADataViewBitSet::ADataViewBitSet(const ADataView& data) : size(data.Size()), bitset(data.GetData()->Size()) {
		for (int k = 0; k < data.NumLabels(); k++) {
			for (auto& instance : data.GetInstancesForLabel(k)) {
				bitset.SetBit(instance->GetID());
			}
		}
	}

	template class Instance<int, ExtraData>;
	template class Instance<double, ExtraData>;

	template void ADataView::TrainTestSplitData<double>(const ADataView&, ADataView&, ADataView&, std::default_random_engine*, double, bool);
	template void ADataView::TrainTestSplitData<int>(const ADataView&, ADataView&, ADataView&, std::default_random_engine*, double, bool);
	template void ADataView::KFoldSplit<double>(const ADataView&, std::vector<ADataView>&, std::vector<ADataView>&, std::default_random_engine*, int, bool);
	template void ADataView::KFoldSplit<int>(const ADataView&, std::vector<ADataView>&, std::vector<ADataView>&, std::default_random_engine*, int, bool);
}