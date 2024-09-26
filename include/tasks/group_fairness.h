#pragma once
#include "tasks/optimization_task.h"

namespace STreeD {

	struct GroupFairnessSol {
		static constexpr double gf_epsilon = 1e-4;
		int misclassifications { 0 };
		double group0_score { 0 };
		double group1_score { 0 };
		bool root_solution { false };

		GroupFairnessSol() = default;
		constexpr GroupFairnessSol(int m, double g0, double g1, bool rs) : misclassifications(m), group0_score(g0), group1_score(g1), root_solution(rs) {}
		constexpr GroupFairnessSol(int m, double g0, double g1) : misclassifications(m), group0_score(g0), group1_score(g1), root_solution(false) {}

		inline bool operator==(const GroupFairnessSol& other) const {
			return other.misclassifications == misclassifications
				&& std::abs(other.group0_score - group0_score) <= gf_epsilon
				&& std::abs(other.group1_score - group1_score) <= gf_epsilon
				&& root_solution == other.root_solution;
		}

		inline bool operator!=(const GroupFairnessSol& other) const { return !(*this == other); }

		inline const GroupFairnessSol& operator+=(const GroupFairnessSol& v2) { 
			misclassifications += v2.misclassifications;
			group0_score += v2.group0_score;
			group1_score += v2.group1_score;
			root_solution = false;
			return *this;
		}

		inline const GroupFairnessSol& operator-=(const GroupFairnessSol& v2) {
			misclassifications = std::max(0, misclassifications - v2.misclassifications);
			group0_score = std::max(0.0, group0_score - v2.group0_score);
			group1_score = std::max(0.0, group1_score - v2.group1_score);
			root_solution = false;
			return *this;
		}

		inline GroupFairnessSol operator+(const GroupFairnessSol& v2) const { return GroupFairnessSol(*this) += v2; }
		inline GroupFairnessSol operator-(const GroupFairnessSol& v2) const { return GroupFairnessSol(*this) -= v2; }

		inline GroupFairnessSol operator*(const int multiplier) { 
			return GroupFairnessSol({ misclassifications * multiplier, group0_score * multiplier, group1_score * multiplier});
		}
	};

	class GroupFairness : public Classification {
	private:
		double discrimination_limit{ 1.0 };
		int train_group0_size{ 0 };
		int train_group1_size{ 0 };
		int test_group0_size{ 0 };
		int test_group1_size{ 0 };
	public:
		using SolType = GroupFairnessSol;
		using SolD2Type = GroupFairnessSol;
		using TestSolType = GroupFairnessSol;

		static const bool total_order = false;
		static const bool custom_leaf = false;
		static const bool check_unique = true;
		static const bool has_constraint = true;
		static const bool terminal_filter = true;
		static const bool terminal_compute_context = true;
		static const bool terminal_zero_costs_true_label = false; // True iff the costs of assigning the true label in the terminal is zero
		static constexpr GroupFairnessSol worst = { INT32_MAX, INT32_MAX, INT32_MAX, false};
		static constexpr GroupFairnessSol best = { 0, 0, 0, true };

		GroupFairness(const ParameterHandler& parameters) : Classification(parameters) {
			discrimination_limit = parameters.GetFloatParameter("discrimination-limit");
		}

		void InformTrainData(const ADataView& train_data, const DataSummary& train_summary);
		void InformTestData(const ADataView& test_data, const DataSummary& test_summary);

		GroupFairnessSol GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const;
		inline GroupFairnessSol GetTestLeafCosts(const ADataView& data, const BranchContext& context, int label) const {
			return GetLeafCosts(data, context, label);
		}
		void GetInstanceLeafD2Costs(const AInstance* instance, int org_label, int label, GroupFairnessSol& costs, int multiplier) const;
		
		inline bool IsD2ZeroCost(const GroupFairnessSol& d2costs) const { return false; }

		void ComputeD2Costs(const GroupFairnessSol& d2costs, int count, GroupFairnessSol& costs) const { costs = d2costs; }
		inline GroupFairnessSol GetWorstPerLabel(int label) const { 
			return label ?
				GroupFairnessSol({ 1, 1.0 / train_group1_size, 1.0 / train_group0_size }) :
				GroupFairnessSol({ 1, 1.0 / train_group0_size, 1.0 / train_group1_size });
		}
		
		double ComputeTrainScore(const GroupFairnessSol& train_value) const;
		double ComputeTrainTestScore(const GroupFairnessSol& test_value) const;
		double ComputeTestTestScore(const GroupFairnessSol& test_value) const;

		inline static GroupFairnessSol Add(const GroupFairnessSol& left, const GroupFairnessSol& right) {
			return left + right;
		}
		inline static GroupFairnessSol TestAdd(const GroupFairnessSol& left, const GroupFairnessSol& right) { return Add(left, right); }
		inline static void Add(const GroupFairnessSol& left, const GroupFairnessSol& right, GroupFairnessSol& out) {
			out = left + right;
		}

		inline static void Subtract(const GroupFairnessSol& left, const GroupFairnessSol& right, GroupFairnessSol& out) {
			out = left - right;
		}

		inline static bool Dominates(const GroupFairnessSol& s1, const GroupFairnessSol& s2) {
			if (s2.root_solution && !s1.root_solution) return false;
			if (s1.root_solution && s1.misclassifications <= s2.misclassifications) return true;
			return (s1.misclassifications <= s2.misclassifications
				&& s1.group0_score <= s2.group0_score + GroupFairnessSol::gf_epsilon
				&& s1.group1_score <= s2.group1_score + GroupFairnessSol::gf_epsilon);
		}

		inline static bool DominatesInv(const GroupFairnessSol& s1, const GroupFairnessSol& s2) {
			return (s1.misclassifications >= s2.misclassifications
				&& s1.group0_score + GroupFairnessSol::gf_epsilon >= s2.group0_score
				&& s1.group1_score + GroupFairnessSol::gf_epsilon >= s2.group1_score);
		}

		inline static bool FrontLT(const GroupFairnessSol& s1, const GroupFairnessSol& s2) {
			throw std::runtime_error("not implemented");
		}

		inline bool DominatesD0(const GroupFairnessSol& s1, const GroupFairnessSol& s2) const {
			return (ComputeTrainScore(s1) >= ComputeTrainScore(s2));
		}

		inline bool DominatesD0Inv(const GroupFairnessSol& s1, const GroupFairnessSol& s2) const {
			return (ComputeTrainScore(s1) <= ComputeTrainScore(s2));
		}

		inline static double CalcDiff(const GroupFairnessSol& s1, const GroupFairnessSol& s2) {
			return (s1.misclassifications - s2.misclassifications) * (s1.misclassifications - s2.misclassifications)
				+ std::abs(s1.group0_score - s2.group0_score) * 100
				+ std::abs(s1.group1_score - s2.group1_score) * 100;
		}

		inline static void Merge(const GroupFairnessSol& s1, const GroupFairnessSol& s2, GroupFairnessSol& out) {
			out = { std::min(s1.misclassifications, s2.misclassifications),
					std::min(s1.group0_score, s2.group0_score),
					std::min(s1.group1_score, s2.group1_score)
				};
		}

		inline static void MergeInv(const GroupFairnessSol& s1, const GroupFairnessSol& s2, GroupFairnessSol& out) {
			out = { std::max(s1.misclassifications, s2.misclassifications),
					std::max(s1.group0_score, s2.group0_score),
					std::max(s1.group1_score, s2.group1_score)
				};
		}

		inline static std::vector<GroupFairnessSol> ExtremePoints() {
			return { {INT32_MAX, 0, 0}, {0, 1, 0}, {0, 0, 1} };
		}

		inline bool SatisfiesConstraint(const Node<GroupFairness>& sol, const BranchContext& context) {
			double disc = std::max(sol.solution.group0_score, sol.solution.group1_score) - 1;
			return disc <= discrimination_limit;
		}

		void RelaxRootSolution(Node<GroupFairness>& sol) const;

		bool MayBranchOnFeature(int feature) const { return feature > 0; }
	};

}

namespace std {
	template <>
	struct hash<STreeD::GroupFairnessSol> {

		//adapted from https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
		size_t operator()(const STreeD::GroupFairnessSol& sol) const {
			using std::size_t;
			using std::hash;
			size_t seed = hash<int>()(sol.misclassifications);
			seed ^= hash<int>()(int(sol.group0_score / STreeD::GroupFairnessSol::gf_epsilon)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= hash<int>()(int(sol.group1_score / STreeD::GroupFairnessSol::gf_epsilon)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			return seed;
		}

	};
}