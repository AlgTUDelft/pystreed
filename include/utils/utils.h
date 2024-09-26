#pragma once
#include "base.h"


namespace STreeD {

	inline int int_log2(unsigned int x) {
		if (x == 0) return INT32_MAX;
		int power = 0;
		while (x >>= 1) { power++; } // compute the log2
		return power;
	}

}