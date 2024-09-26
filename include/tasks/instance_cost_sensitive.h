#pragma once
#include "tasks/optimization_task.h"

namespace STreeD {

	template <class OT>
	struct PairWorstCount;

    struct InstanceCostSensitiveData {

        InstanceCostSensitiveData() : costs() {}

        InstanceCostSensitiveData(std::vector<double>& costs) : costs(costs) {
            worst = *std::max_element(std::begin(costs), std::end(costs));
        }
        
        static InstanceCostSensitiveData ReadData(std::istringstream& iss, int num_labels);

        // Get the cost for classifying this instance with the given label
        inline double GetLabelCost(int label) const { return costs.at(label); }
        
        // Add the label cost to the cost vector
        inline void AddLabelCost(double cost) { costs.push_back(cost); }
        
        // Return the number of possible labels
        inline int NumLabels() const { return int(costs.size()); }
        
        // Get the worst possible score for this instance (max of costs)
		inline double GetWorst() const { return worst; }
    
        // The costs for classifying this instance with each of the labels
        std::vector<double> costs;
		// The worst cost (max of costs)
        double worst{ 0 };
    };

	class InstanceCostSensitive : public Classification {
	public:
		using SolType = double;
		using SolD2Type = double;
		using TestSolType = double;

        using ET = InstanceCostSensitiveData;

		static const bool total_order = true;
		static const bool custom_leaf = false;
		static const bool preprocess_train_test_data = true;
		static const bool custom_similarity_lb = true;
        static const bool terminal_zero_costs_true_label = false; // True iff the costs of assigning the true label in the terminal is zero
		static constexpr  double worst = DBL_MAX;
		static constexpr  double best = 0;

		explicit InstanceCostSensitive(const ParameterHandler& parameters)
            : num_labels(int(parameters.GetIntegerParameter("num-extra-cols"))),
              Classification(parameters) {}

        inline void UpdateParameters(const ParameterHandler& parameters) {
            num_labels = int(parameters.GetIntegerParameter("num-extra-cols"));
        }

        // Compute the leaf costs for the data in the context when assigning label
        double GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const;

        // Compute the test leaf costs for the data in the context when assigning label
		double GetTestLeafCosts(const ADataView& data, const BranchContext& context, int label) const;

        // Compute the leaf costs for an instance given a assigned label
		void GetInstanceLeafD2Costs(const AInstance* instance, int org_label, int label, double& costs, int multiplier) const;
		
        // Compute the solution value from a terminal solution value
        void  ComputeD2Costs(const double& d2costs, int count, double& costs) const;
		
        // Return true if the terminal solution value is zero
        inline bool IsD2ZeroCost(const double& d2costs) const { return d2costs <= 1e-6 && d2costs >= -1e-6; }
		
        // Get a bound on the worst contribution to the objective of a single instance with label
        inline double GetWorstPerLabel(int label) const { return worst_per_label.at(label); }
        
        // Inform the task on what training data is used
        void InformTrainData(const ADataView& train_data, const DataSummary& train_summary);
        
        // Compute the train score from the training solution value
        inline double ComputeTrainScore(double train_value) const { return train_value ; }
        
        // Compute the test score on the training data from the test solution value
        inline double ComputeTrainTestScore(double train_value) const { return train_value; }
        
        // Compute the test score on the test data from the test solution value
        inline double ComputeTestTestScore(double test_value) const { return test_value; }
		
        // Compare two score values. Lower is better
        inline static bool CompareScore(double score1, double score2) { return score1 - score2 <= 0; } // return true if score1 is better than score2

        // Provide a custom lower bound based on the worst values possible per instance
        PairWorstCount<InstanceCostSensitive> ComputeSimilarityLowerBound(const ADataView& data_old, const ADataView& data_new) const;

        // Preprocess the training and test data
        void PreprocessTrainData(ADataView& train_data);
        void PreprocessTestData(ADataView& test_data);

    private:
        std::vector<double> worst_per_label;
        int num_labels{ 1 };
    };

}