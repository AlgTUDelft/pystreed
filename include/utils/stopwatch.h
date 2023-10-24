/**
From Emir Demirovic "MurTree" 
https://bitbucket.org/EmirD/murtree
*/
#pragma once
#include <base.h>

namespace STreeD {

	class Stopwatch {
	public:
		Stopwatch() :starting_time(0), time_limit_in_seconds(0) {}

		void Initialise(double time_limit_in_seconds) {
			starting_time = time(nullptr);
			this->time_limit_in_seconds = time_limit_in_seconds;
		}

		double TimeElapsedInSeconds() const {
			return difftime(time(nullptr), starting_time);
		}

		double TimeLeftInSeconds() const {
			return time_limit_in_seconds - TimeElapsedInSeconds();
		}


		bool IsWithinTimeLimit() const {
			return TimeElapsedInSeconds() < time_limit_in_seconds;
		}


	private:
		time_t starting_time;
		double time_limit_in_seconds;
	};

}