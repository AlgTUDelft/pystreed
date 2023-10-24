#pragma once
#include "base.h"
#include "utils/parameter_handler.h"

namespace STreeD {

	struct TuneRunConfiguration {
		bool reset_solver{ true };
		bool skip_when_max_tree{ false };
		int runs{ 5 };
		double validation_percentage{ 0.2 };
		std::vector<ParameterHandler> parameters;
		std::vector<std::string> descriptors;

		inline int GetNumberOfConfigs() const { return int(parameters.size()); }
		inline int GetNumberOfRuns() const { return runs; }

		inline void AddConfiguration(const ParameterHandler& params, const std::string& descriptor) {
			parameters.push_back(params);
			descriptors.push_back(descriptor);
		}
	};

}