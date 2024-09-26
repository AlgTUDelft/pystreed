/**
 * Optimal Survival Trees, with Dynamic Programming
 * Tim Huisman, Jacobus G. M. van der Linden, Emir Demirovic
*/
#pragma once
#include "tasks/optimization_task.h"
#include <functional>

namespace STreeD {
	
	// Extra data for Survival analysis
	class SAData {

	public:
		static SAData ReadData(std::istringstream& iss, int num_labels);

		SAData() = default;

		inline SAData(int ev, double hazard) {
			this->ev = ev;
			this->hazard = hazard;
		};

		// Get the event (censoring = 0, or death = 1)
		inline int GetEvent() const { return ev; }
		
		// Get the hazard value (estimated by Lambda(t), the Nelson-Aalen estimator)
		inline double GetHazard() const { return hazard; } 
		
		void SetHazard(double hazard) { runtime_assert(hazard > 0); this->hazard = hazard; }

	protected:
		int ev{ 0 };				// The event (censoring = 0, or death = 1)
		double hazard{ -1.0 };		// The hazard value (estimated by Lambda(t), the Nelson-Aalen estimator)
	};

	// Data class to store the sums for the depth-two solver
	struct D2SASol {
		double hazard_sum{ 0 };
		int event_sum{ 0 };
		double negative_log_hazard_sum{ 0 };

		inline const D2SASol& operator+=(const D2SASol& v2) {
			hazard_sum += v2.hazard_sum;
			event_sum += v2.event_sum;
			negative_log_hazard_sum += v2.negative_log_hazard_sum;
			return *this;
		}
		inline D2SASol operator+(const D2SASol& v2) const { return D2SASol(*this) += v2; }
		inline const D2SASol& operator-=(const D2SASol& v2) {
			hazard_sum -= v2.hazard_sum;
			event_sum -= v2.event_sum;
			negative_log_hazard_sum -= v2.negative_log_hazard_sum;
			return *this;
		}
		inline D2SASol operator-(const D2SASol& v2) const { return D2SASol(*this) -= v2; }
		inline bool operator==(const D2SASol& v2) const {
			return std::abs(hazard_sum - v2.hazard_sum) < 1e-6
				&& event_sum == v2.event_sum
				&& std::abs(negative_log_hazard_sum - v2.negative_log_hazard_sum) < 1e-6;
		}
		inline bool operator!=(const D2SASol& v2) const { return !(*this == v2); }
	};

	/*
	 * Optimize survival trees, based on the proportional hazard model from
	 * LeBlanc and Crowly, Relative Risk for Censored Survival Data, Biometrics 48(2), 1992, pp. 411-425.
	 */
	class SurvivalAnalysis : public OptimizationTask {
	private:
		// extra data instances created in preprocessing
		AData train_data_storage, test_data_storage; 
		
		// The Nelson-Aalen estimator used as the baseline cumulative hazard function estimate, computed from the training data
		std::function<double(double)> nelson_aalen; 

	public:
		using ET = SAData;					// The extra data type
		using SolType = double;				// The type of the loss is double
		using SolLabelType = double;		// The type of the theta estimate is double
		using SolD2Type = D2SASol;			// The type of the depth-two solution (three sums)
		using TestSolType = double;			// The type of the test loss is double
		using LabelType = double;			// The type of the label in the data set (time of event)

		static const bool preprocess_train_test_data = true;	// Preprocessing computes the baseline cumulative hazard function
		static const bool use_terminal = true;					// activates the depth-two solver
		static const bool element_additive = false;				// deactivates the similarity lower bound
		static const bool terminal_zero_costs_true_label = false; // True iff the costs of assigning the true label in the terminal is zero

		static const bool total_order = true;			// This otimization task is totally ordered
		static const bool custom_leaf = true;			// A custom leaf node optimization function is provided
		static constexpr  double worst = DBL_MAX;		// The worst solution value (infinite loss)
		static constexpr  double best = 0;				// The best solution value (zero loss)

		SurvivalAnalysis(const ParameterHandler& parameters) { }

		inline void UpdateParameters(const ParameterHandler& parameters) {}

		// Compute the baseline cumulative hazard function from the training data
		static std::function<double(double)> ComputeHazardFunction(const std::vector<const AInstance*>& instances);

		// Apply the hazard function to the data set and store the newly created data in extra_data
		void ApplyHazardFunction(ADataView& dataset, AData& extra_data);

		// Solve a leaf node by finding the max-likelihood estimate for theta with its corresponding loss
		Node<SurvivalAnalysis> SolveLeafNode(const ADataView& data, const ContextType& context) const;
		
		// Get the loss for a leaf node given a theta estimate
		double GetLeafCosts(const ADataView& data, const ContextType& context, double theta) const;
		
		// Get the test loss for a leaf node given a theta estimate
		inline double GetTestLeafCosts(const ADataView& data, const ContextType& context, double theta) const { return GetLeafCosts(data, context, theta); }

		// Classify an instance, return the theta estimate
		double Classify(const AInstance*, double label) const { return label; }
		
		// Get the depth two costs for the given instance (event sum, hazard sum, negative log hazard sum)
		void GetInstanceLeafD2Costs(const AInstance* instance, int org_label, int label, D2SASol& costs, int multiplier) const;
		
		// Compute the loss from the depth-two cost tuple
		void ComputeD2Costs(const D2SASol& d2costs, int count, double& costs) const;

		// Return true if the depth-two contribution is zero (always false)
		inline bool IsD2ZeroCost(const D2SASol& d2costs) const { return false; }
		
		// Compute the max-likelihood theta estimate from the depht-two cost tuple
		double GetLabel(const D2SASol& costs, int count) const;
		
		// Compute the training score (average loss)
		inline double ComputeTrainScore(double train_value) const { return train_value / train_summary.size; }
		
		// Compute the test score on the training data (average loss)
		inline double ComputeTrainTestScore(double train_value) const { return train_value / train_summary.size; }
		
		// Compute the test score on the test data (average loss)
		inline double ComputeTestTestScore(double test_value) const { return test_value / test_summary.size; }
		
		// Compare two scores (lower is better)
		inline static bool CompareScore(double score1, double score2) { return score1 < score2; } // return true if score1 is better than score2
				
		// Preprocess the training data: compute and apply the baseline cumulative hazard function
		void PreprocessTrainData(ADataView& train_data);
		
		// Preprocess the training data: apply the baseline cumulative hazard function
		void PreprocessTestData(ADataView& test_data);

		// Get the configurations for hypertuning
		static TuneRunConfiguration GetTuneRunConfiguration(const ParameterHandler& default_config, const ADataView& train_data, int phase);
	};

}