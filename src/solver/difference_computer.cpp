/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/

#include "solver/difference_computer.h"

namespace STreeD {
	DifferenceMetrics BinaryDataDifferenceComputer::ComputeDifferenceMetrics(const ADataView& data_old, const ADataView& data_new) {
		DifferenceMetrics metrics(data_new.NumLabels());
		
		for (int label = 0; label < data_new.NumLabels(); label++) {
			auto& new_instances = data_new.GetInstancesForLabel(label);
			auto& old_instances = data_old.GetInstancesForLabel(label);
			int size_new = int(new_instances.size());
			int size_old = int(old_instances.size());
			int index_new = 0, index_old = 0;
			while (index_new < size_new && index_old < size_old) {
				int id_new = new_instances[index_new]->GetID();
				int id_old = old_instances[index_old]->GetID();

				//the new data has something the old one does not
				if (id_new < id_old) {
					metrics.total_difference++;
					index_new++;
				}
				//the old data has something the new one does not
				else if (id_new > id_old) {
					metrics.total_difference++;
					metrics.num_removals[label]++;
					index_old++;
				} else {//no difference
					index_new++;
					index_old++;
				}
			}
			metrics.total_difference += (size_new - index_new);
			metrics.total_difference += (size_old - index_old);
			metrics.num_removals[label] += (size_old - index_old);
		}
		return metrics;
	}

	void BinaryDataDifferenceComputer::ComputeDifference(const ADataView& data_old, const ADataView& data_new, ADataView& data_to_add, ADataView& data_to_remove) {

		for (int label = 0; label < data_new.NumLabels(); label++) {
			auto& new_instances = data_new.GetInstancesForLabel(label);
			auto& old_instances = data_old.GetInstancesForLabel(label);
			int size_new = int(new_instances.size());
			int size_old = int(old_instances.size());
			int index_new = 0, index_old = 0;
			int id_new_prev = -1;
			int id_old_prev = -1;
			while (index_new < size_new && index_old < size_old) {
				int id_new = new_instances[index_new]->GetID();
				int id_old = old_instances[index_old]->GetID();
				runtime_assert(id_new_prev <= id_new && id_old_prev <= id_old);
				id_new_prev = id_new;
				id_old_prev = id_old;
				//the new data has something the old one does not
				if (id_new < id_old) {
					data_to_add.AddInstance(label, new_instances[index_new]);
					index_new++;
				}
				//the old data has something the new one does not
				else if (id_new > id_old) {
					data_to_remove.AddInstance(label, old_instances[index_old]);
					index_old++;
				} else {//no difference
					index_new++;
					index_old++;
				}
			}//end while

			for (; index_new < size_new; index_new++) {
				data_to_add.AddInstance(label, new_instances[index_new]);
			}

			for (; index_old < size_old; index_old++) {
				data_to_remove.AddInstance(label, old_instances[index_old]);
			}
		}
	}

}
