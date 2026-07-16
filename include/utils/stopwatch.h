/**
From Emir Demirovic "MurTree" 
https://bitbucket.org/EmirD/murtree
*/
#pragma once
#include <base.h>

namespace STreeD {

	class Stopwatch {
	public:
		Stopwatch() :
			starting_time(std::chrono::steady_clock::now()),
			time_limit_in_seconds(0) {}

		void Initialise(double time_limit_in_seconds) {
			starting_time = std::chrono::steady_clock::now();
			this->time_limit_in_seconds = time_limit_in_seconds;
			this->enabled = true;
		}

		double TimeElapsedInSeconds() const {
			return double(std::chrono::duration_cast<std::chrono::microseconds>(
				std::chrono::steady_clock::now() - starting_time
			).count()) / 1e6;
		}

		double TimeLeftInSeconds() const {
			return time_limit_in_seconds - TimeElapsedInSeconds();
		}


		bool IsWithinTimeLimit() const {
			return !enabled || TimeElapsedInSeconds() < time_limit_in_seconds;
		}

		void Enable() { enabled = true; }
		void Disable() { enabled = false; }

	private:
		std::chrono::steady_clock::time_point starting_time;
		double time_limit_in_seconds;
		bool enabled{ true };
	};

}