#pragma once
#include "tasks/optimization_task.h"

namespace STreeD {

	// A custom solution class for F1-score
	// counting the false positives and negatives
	struct F1ScoreSol {
		int false_negatives{ 0 };
		int false_positives{ 0 };

		inline bool operator==(const F1ScoreSol& other) const {
			return other.false_negatives == false_negatives &&
				other.false_positives == false_positives;
		}

		inline bool operator!=(const F1ScoreSol& other) const { return !(*this == other); }

		inline const F1ScoreSol& operator+=(const F1ScoreSol& v2) { false_negatives += v2.false_negatives; false_positives += v2.false_positives; return *this; }
		inline const F1ScoreSol& operator-=(const F1ScoreSol& v2) { false_negatives -= v2.false_negatives; false_positives -= v2.false_positives; return *this; }
		inline F1ScoreSol operator+(const F1ScoreSol& v2) const { return F1ScoreSol(*this) += v2; }
		inline F1ScoreSol operator-(const F1ScoreSol& v2) const { return F1ScoreSol(*this) -= v2; }

		inline F1ScoreSol operator*(const int multiplier) { return F1ScoreSol({ false_negatives * multiplier, false_positives * multiplier }); }
	};

	class F1Score : public Classification {
	public:
		using SolType = F1ScoreSol;
		using SolD2Type = F1ScoreSol;
		using TestSolType = F1ScoreSol;

		static const bool total_order = false;
		static const bool custom_leaf = false;
		static const bool check_unique = false;
		static const bool terminal_zero_costs_true_label = false; // True iff the costs of assigning the true label in the terminal is zero
		static constexpr F1ScoreSol worst = { INT32_MAX, INT32_MAX };
		static constexpr F1ScoreSol best = { 0, 0 };

		F1Score(const ParameterHandler& parameters) : Classification(parameters) {}

		F1ScoreSol GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const;
		inline F1ScoreSol GetTestLeafCosts(const ADataView& data, const BranchContext& context, int label) const {
			return GetLeafCosts(data, context, label);
		}
		inline void GetInstanceLeafD2Costs(const AInstance* instance, int org_label, int label, F1ScoreSol& costs, int multiplier) const {
			costs = label == 0
				? F1ScoreSol({ org_label == 1 ? multiplier : 0, 0 })
				: F1ScoreSol({ 0, org_label == 0 ? multiplier : 0 });
		}
		inline bool IsD2ZeroCost(const F1ScoreSol& d2costs) const { return d2costs.false_negatives == 0 && d2costs.false_positives == 0; }
		void ComputeD2Costs(const F1ScoreSol& d2costs, int count, F1ScoreSol& costs) const { costs = d2costs; }
		inline F1ScoreSol GetWorstPerLabel(int label) const { return label == 0 ? F1ScoreSol({ 0, 1 }) : F1ScoreSol({ 1, 0 }); }
		
		double ComputeTrainScore(const F1ScoreSol& train_value) const;
		double ComputeTrainTestScore(const F1ScoreSol& test_value) const;
		double ComputeTestTestScore(const F1ScoreSol& test_value) const;

		inline static F1ScoreSol Add(const F1ScoreSol& left, const F1ScoreSol& right) {
			return { left.false_negatives + right.false_negatives, left.false_positives + right.false_positives };
		}
		inline static F1ScoreSol TestAdd(const F1ScoreSol& left, const F1ScoreSol& right) { return Add(left, right); }
		inline static void Add(const F1ScoreSol& left, const F1ScoreSol& right, F1ScoreSol& out) {
			out.false_negatives = left.false_negatives + right.false_negatives;
			out.false_positives = left.false_positives + right.false_positives;
		}

		inline static void Subtract(const F1ScoreSol& left, const F1ScoreSol& right, F1ScoreSol& out) {
			out = F1ScoreSol({ std::max(0, left.false_negatives - right.false_negatives),
								std::max(0, left.false_positives - right.false_positives) });
		}

		inline static bool Dominates(const F1ScoreSol& s1, const F1ScoreSol& s2) {
			return (s1.false_negatives <= s2.false_negatives && s1.false_positives <= s2.false_positives);

		}

		inline static bool DominatesInv(const F1ScoreSol& s1, const F1ScoreSol& s2) {
			return (s1.false_negatives >= s2.false_negatives && s1.false_positives >= s2.false_positives);
		}

		// (currently not used)
		inline static bool FrontLT(const F1ScoreSol& s1, const F1ScoreSol& s2) {
			return (s1.false_positives < s2.false_positives || (s1.false_positives == s2.false_positives && s1.false_negatives < s2.false_negatives));
		}

		inline bool DominatesD0(const F1ScoreSol& s1, const F1ScoreSol& s2) const {
			return (ComputeTrainScore(s1) >= ComputeTrainScore(s2));
		}

		inline bool DominatesD0Inv(const F1ScoreSol& s1, const F1ScoreSol& s2) const {
			return (ComputeTrainScore(s1) <= ComputeTrainScore(s2));
		}

		inline static double CalcDiff(const F1ScoreSol& s1, const F1ScoreSol& s2) {
			return (s1.false_negatives - s2.false_negatives) * (s1.false_negatives - s2.false_negatives)
				+ (s1.false_positives - s2.false_positives) * (s1.false_positives - s2.false_positives);
		}

		inline static void Merge(const F1ScoreSol& s1, const F1ScoreSol& s2, F1ScoreSol& out) {
			out = { std::min(s1.false_negatives, s2.false_negatives), std::min(s1.false_positives, s2.false_positives) };
		}

		inline static void MergeInv(const F1ScoreSol& s1, const F1ScoreSol& s2, F1ScoreSol& out) {
			out = { std::max(s1.false_negatives, s2.false_negatives), std::max(s1.false_positives, s2.false_positives) };
		}

		inline static std::vector<F1ScoreSol> ExtremePoints() {
			return { {0, INT32_MAX}, {INT32_MAX, 0} };
		}
	};

}

namespace std {
	template <>
	struct hash<STreeD::F1ScoreSol> {

		//adapted from https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
		size_t operator()(const STreeD::F1ScoreSol& sol) const {
			using std::size_t;
			using std::hash;
			size_t seed = hash<int>()(sol.false_negatives);
			seed ^= hash<int>()(sol.false_positives) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			return seed;
		}
	};

}