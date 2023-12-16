#include "tasks/survival_analysis.h"

namespace STreeD {

	// Read the event (censoring = 0 or death = 1) from the file. Set the hazard value to -1 temporarily
	SAData SAData::ReadData(std::istringstream& iss, int num_labels) {
		int ev;
		iss >> ev;
		return { ev, -1.0 };
	}

	// Apply the hazard function to the data set and store the newly created data in extra_data
	void SurvivalAnalysis::ApplyHazardFunction(ADataView& dataset, AData& extra_data) {
		// Clear any data in extra data
		extra_data.Clear();
		
		auto& instances = dataset.GetMutableInstancesForLabel(0);
		for (int i = 0; i < instances.size(); i++) {
			// Create a copy of the instance and assign it the hazard value compute with the nelson-aalen estimator
			auto instance = static_cast<const Instance<double, SAData>*>(instances[i]);
			auto copy_instance = new Instance<double, SAData>(*static_cast<const Instance<double, SAData>*>(instance));
			double hazard = nelson_aalen(copy_instance->GetLabel());
			copy_instance->GetMutableExtraData().SetHazard(hazard);
			instances[i] = copy_instance; // Replace old with new instance
			extra_data.AddInstance(copy_instance);
		}
	}
	
	// Preprocess the training data: compute and apply the baseline cumulative hazard function
	void SurvivalAnalysis::PreprocessTrainData(ADataView& train_data) {
		nelson_aalen = ComputeHazardFunction(train_data.GetInstancesForLabel(0));
		ApplyHazardFunction(train_data, train_data_storage);
	}

	// Preprocess the training data: apply the baseline cumulative hazard function
	void SurvivalAnalysis::PreprocessTestData(ADataView& test_data) {
		ApplyHazardFunction(test_data, test_data_storage);
	}

	// Compute the baseline cumulative hazard function from the training data using the Nelson Aalen estimator
	std::function<double (double)> SurvivalAnalysis::ComputeHazardFunction(const std::vector<const AInstance*>& instances) {
		std::map<double, std::pair<int, int>> events_per_time = {};
		std::vector<double> times = {};

		int at_risk = 0;
		for (const auto i : instances) {
			auto instance = static_cast<const Instance<double, SAData>*>(i);
			double time = instance->GetLabel();
			int event = instance->GetExtraData().GetEvent();

			if (events_per_time.find(time) == events_per_time.end()) {
				events_per_time[time] = std::pair<int, int>{ 0, 0 };
				times.push_back(time);
			}

			if (event == 0) {
				events_per_time[time].first++;
			}
			else {
				events_per_time[time].second++;
			}

			at_risk++;
		}

		std::sort(times.begin(), times.end());

		std::vector<double> hazard_keys = {};
		std::vector<double> hazard_values = {};
		double sum = 0.0;
		for (double t : times) {
			int censored = events_per_time[t].first;
			int died = events_per_time[t].second;

			sum += (double)died / (double)at_risk;
			at_risk -= died + censored;

			if (died == 0)
				continue;

			hazard_keys.push_back(t);
			hazard_values.push_back(sum);
		}

		// When no earlier event is recorded, return 1/(n+1) (not zero, to prevent log(0))
		hazard_keys.insert(hazard_keys.begin(), 0.0);
		hazard_values.insert(hazard_values.begin(), 1.0 / (instances.size() + 1));

		auto hazard_function = [hazard_keys, hazard_values](double t) {
			int a = 0;
			int b = int(hazard_values.size()) - 1;

			while (a != b) {
				int mid = (a + b + 1) / 2;

				if (hazard_keys[mid] > t + 1e-6)
					b = mid - 1;
				else
					a = mid;
			}

			return hazard_values[a];
		};

		return hazard_function;
	}

	// Solve a leaf node by finding the max-likelihood estimate for theta with its corresponding loss
	Node<SurvivalAnalysis> SurvivalAnalysis::SolveLeafNode(const ADataView& data, const ContextType& context) const {
		// First compute the three sums
		double hazard_sum = 0;
		int event_sum = 0;
		double negative_log_hazard_sum = 0;

		for (const auto i : data.GetInstancesForLabel(0)) {
			auto instance = static_cast<const Instance<double, SAData>*>(i);
			int ev = instance->GetExtraData().GetEvent();
			double hazard = instance->GetExtraData().GetHazard();
			runtime_assert(!ev || hazard > 0);

			hazard_sum += hazard;
			if (ev) {
				event_sum += 1;
				negative_log_hazard_sum += -log(hazard);
			}
		}

		// Compute the theta estimate and the coressponding loss
		double theta;
		if (event_sum == 0) {
			// To prevent theta = 0, consider the number of observed deaths = 0.5 See LeBlanc and Crowly, p. 416
			theta = 0.5 / hazard_sum;
		} else {
			theta = event_sum / hazard_sum;
		}
		double error = negative_log_hazard_sum - event_sum * log(theta);
		runtime_assert(error >= -1e-6);
		return Node<SurvivalAnalysis>(theta, std::max(0.0, error));
	}

	// Get the loss for a leaf node given a theta estimate
	double SurvivalAnalysis::GetLeafCosts(const ADataView& data, const ContextType& context, double theta) const {
		// First compute the three sums
		double hazard_sum = 0;
		int event_sum = 0;
		double negative_log_hazard_sum = 0;

		for (const auto i : data.GetInstancesForLabel(0)) {
			auto instance = static_cast<const Instance<double, SAData>*>(i);
			int ev = instance->GetExtraData().GetEvent();
			double hazard = instance->GetExtraData().GetHazard();
			runtime_assert(!ev || hazard > 0);

			hazard_sum += hazard;
			if (ev) {
				event_sum += 1;
				negative_log_hazard_sum += -log(hazard);
			}
		}
		
		// Compute error based on a given theta
		double error = negative_log_hazard_sum - event_sum * (log(theta) + 1) + theta * hazard_sum;
		runtime_assert(error >= -1e-6);
		return std::max(0.0, error);
	}

	// Compute the contribution of the given instance to the depth-two cost tuple
	void SurvivalAnalysis::GetInstanceLeafD2Costs(const AInstance* ainstance, int org_label, int label, D2SASol& costs, int multiplier) const {
		auto instance = static_cast<const Instance<double, SAData>*>(ainstance);
		int ev = instance->GetExtraData().GetEvent();
		double hazard = instance->GetExtraData().GetHazard();

		costs.hazard_sum = hazard * multiplier;
		if (ev) {
			costs.event_sum = multiplier;
			costs.negative_log_hazard_sum = -log(hazard) * multiplier;
		}
		else {
			costs.event_sum = 0;
			costs.negative_log_hazard_sum = 0;
		}
	}

	// Compute the loss from the given depth-two cost tuple
	void SurvivalAnalysis::ComputeD2Costs(const D2SASol& d2costs, int count, double& costs) const {
		costs = std::max(0.0, d2costs.negative_log_hazard_sum - d2costs.event_sum * log(std::max(0.5, double(d2costs.event_sum)) / d2costs.hazard_sum));
	}

	// Get the mak-likelihood theta estimate for the given depth-two cost tuple
	double SurvivalAnalysis::GetLabel(const D2SASol& costs, int count) const {
		runtime_assert(costs.hazard_sum > 0);
		return std::max(0.5, double(costs.event_sum)) / costs.hazard_sum;
	}

	TuneRunConfiguration SurvivalAnalysis::GetTuneRunConfiguration(const ParameterHandler& default_config, const ADataView& data, int phase) {
		TuneRunConfiguration config = OptimizationTask::GetTuneRunConfiguration(default_config, data, phase);
		config.runs = 10;
		return config;
	}

}