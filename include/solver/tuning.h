#pragma once
#include "base.h"
#include "utils/parameter_handler.h"
#include "solver/result.h"

namespace STreeD {

	struct TuneRunConfiguration {
		bool reset_solver{ true };
		bool skip_when_max_tree{ false };
		std::vector<ParameterHandler> parameters;
		std::vector<std::string> descriptors;
		std::vector<std::shared_ptr<SolverResult>> results;

		inline int GetNumberOfConfigs() const { return int(parameters.size()); }
		inline bool HasResults() const { return results.size() > 0; }

		inline void AddConfiguration(const ParameterHandler& params, const std::string& descriptor) {
			parameters.push_back(params);
			descriptors.push_back(descriptor);
		}

		inline void AddResult(const std::shared_ptr<SolverResult>& result) {
			results.push_back(result);
		}
	};

}