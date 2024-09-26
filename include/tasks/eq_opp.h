#pragma once
#include "tasks/optimization_task.h"

namespace STreeD {

	struct EqOppSol {
		static constexpr double gf_epsilon = 1e-4;
		int misclassifications{ 0 };
		double group0_score{ 0 };
		double group1_score{ 0 };
		bool root_solution{ false };

		EqOppSol() = default;
		constexpr EqOppSol(int m, double g0, double g1, bool rs) : misclassifications(m), group0_score(g0), group1_score(g1), root_solution(rs) {}
		constexpr EqOppSol(int m, double g0, double g1) : misclassifications(m), group0_score(g0), group1_score(g1), root_solution(false) {}

		inline bool operator==(const EqOppSol& other) const {
			return other.misclassifications == misclassifications
				&& std::abs(other.group0_score - group0_score) <= gf_epsilon
				&& std::abs(other.group1_score - group1_score) <= gf_epsilon
				&& root_solution == other.root_solution;
		}

		inline bool operator!=(const EqOppSol& other) const { return !(*this == other); }

		inline const EqOppSol& operator+=(const EqOppSol& v2) {
			misclassifications += v2.misclassifications;
			group0_score += v2.group0_score;
			group1_score += v2.group1_score;
			root_solution = false;
			return *this;
		}

		inline const EqOppSol& operator-=(const EqOppSol& v2) {
			misclassifications = std::max(0, misclassifications - v2.misclassifications);
			group0_score = std::max(0.0, group0_score - v2.group0_score);
			group1_score = std::max(0.0, group1_score - v2.group1_score);
			root_solution = false;
			return *this;
		}

		inline EqOppSol operator+(const EqOppSol& v2) const { return EqOppSol(*this) += v2; }
		inline EqOppSol operator-(const EqOppSol& v2) const { return EqOppSol(*this) -= v2; }

		inline EqOppSol operator*(const int multiplier) {
			return EqOppSol({ misclassifications * multiplier, group0_score * multiplier, group1_score * multiplier });
		}
	};

	class EqOpp : public Classification {
	private:
		double discrimination_limit{ 1.0 };
		int train_group0_size{ 0 };
		int train_group1_size{ 0 };
		int test_group0_size{ 0 };
		int test_group1_size{ 0 };
	public:
		using SolType = EqOppSol;
		using SolD2Type = EqOppSol;
		using TestSolType = EqOppSol;

		static const bool total_order = false;
		static const bool custom_leaf = false;
		static const bool check_unique = true;
		static const bool has_constraint = true;
		static const bool terminal_filter = true;
		static const bool terminal_compute_context = true;
		static const bool terminal_zero_costs_true_label = false; // True iff the costs of assigning the true label in the terminal is zero
		static constexpr EqOppSol worst = { INT32_MAX, INT32_MAX, INT32_MAX, false };
		static constexpr EqOppSol best = { 0, 0, 0, true };

		EqOpp(const ParameterHandler& parameters) : Classification(parameters) {
			discrimination_limit = parameters.GetFloatParameter("discrimination-limit");
		}

		void InformTrainData(const ADataView& train_data, const DataSummary& train_summary);
		void InformTestData(const ADataView& test_data, const DataSummary& test_summary);

		EqOppSol GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const;
		inline EqOppSol GetTestLeafCosts(const ADataView& data, const BranchContext& context, int label) const {
			return GetLeafCosts(data, context, label);
		}
		void GetInstanceLeafD2Costs(const AInstance* instance, int org_label, int label, EqOppSol& costs, int multiplier) const;

		inline bool IsD2ZeroCost(const EqOppSol& d2costs) const { return false; }

		void ComputeD2Costs(const EqOppSol& d2costs, int count, EqOppSol& costs) const { costs = d2costs; }
		inline EqOppSol GetWorstPerLabel(int label) const {
			return label ?
				EqOppSol({ 1, 1.0 / train_group1_size, 1.0 / train_group0_size }) :
				EqOppSol({ 1, 0.0, 0.0 });
		}

		double ComputeTrainScore(const EqOppSol& train_value) const;
		double ComputeTrainTestScore(const EqOppSol& test_value) const;
		double ComputeTestTestScore(const EqOppSol& test_value) const;

		inline static EqOppSol Add(const EqOppSol& left, const EqOppSol& right) {
			return left + right;
		}
		inline static EqOppSol TestAdd(const EqOppSol& left, const EqOppSol& right) { return Add(left, right); }
		inline static void Add(const EqOppSol& left, const EqOppSol& right, EqOppSol& out) {
			out = left + right;
		}

		inline static void Subtract(const EqOppSol& left, const EqOppSol& right, EqOppSol& out) {
			out = left - right;
		}

		inline static bool Dominates(const EqOppSol& s1, const EqOppSol& s2) {
			if (s2.root_solution && !s1.root_solution) return false;
			if (s1.root_solution && s1.misclassifications <= s2.misclassifications) return true;
			return (s1.misclassifications <= s2.misclassifications
				&& s1.group0_score <= s2.group0_score + EqOppSol::gf_epsilon
				&& s1.group1_score <= s2.group1_score + EqOppSol::gf_epsilon);
		}

		inline static bool DominatesInv(const EqOppSol& s1, const EqOppSol& s2) {
			return (s1.misclassifications >= s2.misclassifications
				&& s1.group0_score + EqOppSol::gf_epsilon >= s2.group0_score
				&& s1.group1_score + EqOppSol::gf_epsilon >= s2.group1_score);
		}

		inline static bool FrontLT(const EqOppSol& s1, const EqOppSol& s2) {
			throw std::runtime_error("not implemented");
		}

		inline bool DominatesD0(const EqOppSol& s1, const EqOppSol& s2) const {
			return (ComputeTrainScore(s1) >= ComputeTrainScore(s2));
		}

		inline bool DominatesD0Inv(const EqOppSol& s1, const EqOppSol& s2) const {
			return (ComputeTrainScore(s1) <= ComputeTrainScore(s2));
		}

		inline static double CalcDiff(const EqOppSol& s1, const EqOppSol& s2) {
			return (s1.misclassifications - s2.misclassifications) * (s1.misclassifications - s2.misclassifications)
				+ std::abs(s1.group0_score - s2.group0_score) * 100
				+ std::abs(s1.group1_score - s2.group1_score) * 100;
		}

		inline static void Merge(const EqOppSol& s1, const EqOppSol& s2, EqOppSol& out) {
			out = { std::min(s1.misclassifications, s2.misclassifications),
					std::min(s1.group0_score, s2.group0_score),
					std::min(s1.group1_score, s2.group1_score)
			};
		}

		inline static void MergeInv(const EqOppSol& s1, const EqOppSol& s2, EqOppSol& out) {
			out = { std::max(s1.misclassifications, s2.misclassifications),
					std::max(s1.group0_score, s2.group0_score),
					std::max(s1.group1_score, s2.group1_score)
			};
		}

		inline static std::vector<EqOppSol> ExtremePoints() {
			return { {INT32_MAX, 0, 0}, {0, 1, 0}, {0, 0, 1} };
		}

		inline bool SatisfiesConstraint(const Node<EqOpp>& sol, const BranchContext& context) {
			double disc = std::max(sol.solution.group0_score, sol.solution.group1_score) - 1;
			return disc <= discrimination_limit;
		}

		void RelaxRootSolution(Node<EqOpp>& sol) const;

		bool MayBranchOnFeature(int feature) const { return feature > 0; }
	};

}

namespace std {
	template <>
	struct hash<STreeD::EqOppSol> {

		//adapted from https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
		size_t operator()(const STreeD::EqOppSol& sol) const {
			using std::size_t;
			using std::hash;
			size_t seed = hash<int>()(sol.misclassifications);
			seed ^= hash<int>()(int(sol.group0_score / STreeD::EqOppSol::gf_epsilon)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= hash<int>()(int(sol.group1_score / STreeD::EqOppSol::gf_epsilon)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			return seed;
		}

	};
}