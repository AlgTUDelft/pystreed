#include "tasks/instance_cost_sensitive.h"
#include "solver/similarity_lowerbound.h"

namespace STreeD {

    void InstanceCostSensitive::PreprocessTrainData(ADataView& train_data) {
        if(num_labels > train_data.NumLabels()) train_data.GetMutableInstances().resize(num_labels);
    }

    void InstanceCostSensitive::PreprocessTestData(ADataView& test_data) {
        if (num_labels > test_data.NumLabels()) test_data.GetMutableInstances().resize(num_labels);
    }

    double InstanceCostSensitive::GetLeafCosts(const ADataView &data, const BranchContext &context, int label) const {
        double costs = 0;
        for (int k = 0; k < data.NumLabels(); k++) {
            for (const auto i : data.GetInstancesForLabel(k)) {
                auto& et = GetInstanceExtraData<int, InstanceCostSensitiveData>(i);
                costs += et.GetLabelCost(label);
            }
        }
        runtime_assert(costs >= 0.0);
        return costs;
    }

    double InstanceCostSensitive::GetTestLeafCosts(const ADataView &data, const BranchContext &context, int label) const {
        double s_costs = GetLeafCosts(data, context, label);
        return s_costs;
    }

    void InstanceCostSensitive::GetInstanceLeafD2Costs(const AInstance *instance, int org_label, int label, double&costs, int multiplier) const {
        auto& et = GetInstanceExtraData<int, InstanceCostSensitiveData>(instance);
        costs = multiplier * et.GetLabelCost(label);
    }

    void InstanceCostSensitive::ComputeD2Costs(const double& d2costs, int count, double &costs) const {
        costs = d2costs < 0 ? 0.0 : d2costs;
    }

    void InstanceCostSensitive::InformTrainData(const ADataView& train_data, const DataSummary& train_summary) {
        OptimizationTask::InformTrainData(train_data, train_summary);
        worst_per_label.clear();
        auto n_labels = train_data.NumLabels();
        for (int i = 0; i < n_labels; i++) {
            worst_per_label.push_back(0);
        }
        for (int i = 0; i < n_labels; i++) {
            const std::vector<const AInstance*>& instances = train_data.GetInstancesForLabel(i);
            for (auto inst : instances) {
                for (int l = 0; l < n_labels; l++) {
                    double current = GetInstanceExtraData<int, InstanceCostSensitiveData>(inst).GetLabelCost(l);
                    if (current > worst_per_label[l]) worst_per_label[l] = current;

                }
            }
        }
    }

    PairWorstCount<InstanceCostSensitive> InstanceCostSensitive::ComputeSimilarityLowerBound(const ADataView& data_old, const ADataView& data_new) const {
        int total_diff = 0;
        double worst_diff = 0;
        
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
                    total_diff += 1;
                    index_new++;
                }
                //the old data has something the new one does not
                else if (id_new > id_old) {
                    total_diff += 1;
                    worst_diff += GetInstanceExtraData<int, InstanceCostSensitiveData>(old_instances[index_old]).GetWorst();
                    index_old++;
                } else {//no difference
                    index_new++;
                    index_old++;
                }
            }
            if (index_new < size_new) total_diff += size_new - index_new;
            for (; index_old < size_old; index_old++) {
                total_diff += 1;
                worst_diff += GetInstanceExtraData<int, InstanceCostSensitiveData>(old_instances[index_old]).GetWorst();
            }
        }
        PairWorstCount<InstanceCostSensitive> result(worst_diff, total_diff);
        return result;
    }


    InstanceCostSensitiveData InstanceCostSensitiveData::ReadData(std::istringstream &iss, int num_labels) {
        InstanceCostSensitiveData extra{};

        double tmp;
        for (int i = 0; i < num_labels; i++) {
            iss >> tmp;
            extra.AddLabelCost(tmp);
        }
        extra.worst = -DBL_MAX;
        for (int i = 0; i < num_labels; i++) {
            if (extra.GetLabelCost(i) > extra.worst)
                extra.worst = extra.GetLabelCost(i);
        }
        return extra;
    }

}