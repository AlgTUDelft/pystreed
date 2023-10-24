#include "tasks/prescriptive_policy.h"

namespace STreeD {

	PPGData PPGData::ReadData(std::istringstream& iss, int num_labels) {
		PPGData ppg;
		iss >> ppg.k;
		iss >> ppg.y;
		iss >> ppg.mu;
		double tmp;
		for (int k = 0; k < num_labels; k++) {
			iss >> tmp;
			ppg.yhat.push_back(tmp);
		}
		iss >> ppg.k_opt;
		for (int k = 0; k < num_labels; k++) {
			iss >> tmp;
			ppg.cf_y.push_back(tmp);
		}
		ppg.ProcessData();
		return ppg;
	}

	void PPGData::ProcessData() {
		int num_labels = int(yhat.size());
		for (int k = 0; k < num_labels; k++) {
			double dm_score = yhat[k];
			double ipw_score = std::abs(mu) <= 1e-6 ? 0 : (k == this->k ? y / mu : 0);
			double dr_score = std::abs(mu) <= 1e-6 ? 0 : (yhat[k] + (k == this->k ? (y - yhat[this->k]) / mu : 0));
			dm_scores.push_back(-dm_score); // Negative because maximization
			ipw_scores.push_back(-ipw_score);
			dr_scores.push_back(-dr_score);
		}
	}


	PrescriptivePolicy::PrescriptivePolicy(const ParameterHandler& parameters) : Classification(parameters) {
		std::string ppg_method = parameters.GetStringParameter("ppg-teacher-method");
		if (ppg_method == "DM") use_dm = true;
		else if (ppg_method == "IPW") use_ipw = true;
		else use_dr = true;
	}

	void PrescriptivePolicy::InformTrainData(const ADataView& train_data, const DataSummary& train_summary) {
		OptimizationTask::InformTrainData(train_data, train_summary);

		int max_id = train_data.Size();
		// Calculate minimum values, so that leaf costs can always be zero or larger
		min_ipw = 1, min_dm = 1, min_dr = 1;
		for (int k = 0; k < train_data.NumLabels(); k++) {
			worst_per_label.push_back(-INT32_MAX);
			for (auto& i : train_data.GetInstancesForLabel(k)) {
				auto instance = static_cast<const Instance<double, PPGData>*>(i);
				auto& ed = instance->GetExtraData();
				for (int k2 = 0; k2 < train_data.NumLabels(); k2++) {
					min_dm = std::min(ed.dm_scores[k2], min_dm);
					min_ipw = std::min(ed.ipw_scores[k2], min_ipw);
					min_dr = std::min(ed.dr_scores[k2], min_dr);
					if (use_dm && ed.dm_scores[k2] > worst_per_label[k]) worst_per_label[k] = ed.dm_scores[k2];
					if (use_ipw && ed.ipw_scores[k2] > worst_per_label[k]) worst_per_label[k] = ed.ipw_scores[k2];
					if (use_dr && ed.dr_scores[k2] > worst_per_label[k]) worst_per_label[k] = ed.dr_scores[k2];
				}
				max_id = std::max(max_id, i->GetID() + 1);
			}
		}
		
		// Calculate the worst case for the similarity lower bound
		for (int k = 0; k < train_data.NumLabels(); k++) {
			if (use_dm) worst_per_label[k] -= min_dm;
			if (use_ipw) worst_per_label[k] -= min_ipw;
			if (use_dr) worst_per_label[k] -= min_dr;
		}

		// Calculate the costs per training instance
		
		cost_per_training_instance = std::vector<std::vector<double>>(max_id, std::vector<double>(train_data.NumLabels(), 0));
		for (int k = 0; k < train_data.NumLabels(); k++) {
			for (auto& i : train_data.GetInstancesForLabel(k)) {
				auto instance = static_cast<const Instance<double, PPGData>*>(i);
				auto& ed = instance->GetExtraData();
				for (int k2 = 0; k2 < train_data.NumLabels(); k2++) {
					if (use_dm) {
						cost_per_training_instance[i->GetID()][k2] = ed.dm_scores[k2] - min_dm;
					} else if (use_ipw) {
						cost_per_training_instance[i->GetID()][k2] = ed.ipw_scores[k2] - min_ipw;
					} else {
						cost_per_training_instance[i->GetID()][k2] = ed.dr_scores[k2] - min_dr;
					}
				}
			}
		}
	}

	double PrescriptivePolicy::GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const {
		double result = 0;
		for (int k = 0; k < data.NumLabels(); k++) {
			for (auto& i : data.GetInstancesForLabel(k)) {
				result += cost_per_training_instance[i->GetID()][label];
			}
		}
		return result;
	}

	int PrescriptivePolicy::GetTestLeafCosts(const ADataView& data, const BranchContext& context, int label) const {
		// Measure out-of-sample correct
		int error = 0;
		for (int k = 0; k < data.NumLabels(); k++) {
			for (auto& i : data.GetInstancesForLabel(k)) {
				auto instance = static_cast<const Instance<double, PPGData>*>(i);
				auto& ed = instance->GetExtraData();
				error += ed.k_opt == label ? 0 : 1;
			}
		}
		return error;
	}

	void PrescriptivePolicy::GetInstanceLeafD2Costs(const AInstance* i, int org_label, int label, double& costs, int multiplier) const {
		auto instance = static_cast<const Instance<double, PPGData>*>(i);
		auto& ed = instance->GetExtraData();
		costs = multiplier * cost_per_training_instance[i->GetID()][label];
	}

}