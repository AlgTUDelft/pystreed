#include "tasks/tasks.h"

namespace STreeD {

	void OptimizationTask::InformTrainData(const ADataView& train_data, const DataSummary& train_summary) {
		this->train_summary = train_summary;
	}

	void OptimizationTask::InformTestData(const ADataView& test_data, const DataSummary& test_summary) {
		this->test_summary = test_summary;
	}

	void OptimizationTask::GetLeftContext(const ADataView& data, const BranchContext& context, int feature, BranchContext& left_context) const {
		Branch::LeftChildBranch(context.GetBranch(), feature, left_context.GetMutableBranch());
	}

	void OptimizationTask::GetRightContext(const ADataView& data, const BranchContext& context, int feature, BranchContext& right_context) const {
		Branch::RightChildBranch(context.GetBranch(), feature, right_context.GetMutableBranch());
	}

	TuneRunConfiguration OptimizationTask::GetTuneRunConfiguration(const ParameterHandler& default_config, const ADataView& data, int phase) {
		TuneRunConfiguration config;

		int max_nodes = int(default_config.GetIntegerParameter("max-num-nodes"));
		int max_d = int(default_config.GetIntegerParameter("max-depth"));

		for (int d = 0; d <= max_d; d++) {
			int _max_nodes = std::min(max_nodes, (1 << d) - 1);
			for (int i = d; i <= _max_nodes; i++) {
				ParameterHandler params = default_config;
				params.SetIntegerParameter("max-depth", d);
				params.SetIntegerParameter("max-num-nodes", i);
				config.AddConfiguration(params, "d=" + std::to_string(d) + ", n=" + std::to_string(i));
			}
		}
		config.reset_solver = false;
		return config;
	}

}

