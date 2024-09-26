#include <stdio.h>
#include <stdlib.h>
#include "base.h"
#include "solver/solver.h"

#include <filesystem>

#ifdef DEBUG
#define debug_assert(x) { assert(x); }
#else
#define debug_assert(x) {}
#endif

#define test_assert(x) {if (!(x)) { printf(#x); debug_assert(x); exit(1); }}
#define test_assert_m(x, m) {if (!(x)) { printf(m); debug_assert(x); exit(1); }}
#define test_failed(m) { printf(m); debug_assert(1==0); exit(1); }

using namespace STreeD;

struct SolverSetup {
	bool use_branch_caching{ true };
	bool use_dataset_caching{ true };
	bool use_upper_bound{ true };
	bool use_lower_bound{ true };
	std::string feature_ordering{ "gini" };

	std::string ToString() const {
		std::ostringstream os;
		os << "Solver setup. branch-cache: " << use_branch_caching << ", dataset-cache: " << use_dataset_caching 
			<< ", upper-bound: " << use_upper_bound << ", lower-bound: " << use_lower_bound
			<< ", feature-ordering: " << feature_ordering;
		return os.str();
	}

	void Apply(ParameterHandler& params) const {
		params.SetBooleanParameter("use-branch-caching", use_branch_caching);
		params.SetBooleanParameter("use-dataset-caching", use_dataset_caching);
		params.SetBooleanParameter("use-upper-bound", use_upper_bound);
		params.SetBooleanParameter("use-lower-bound", use_lower_bound);
		params.SetStringParameter("feature-ordering", feature_ordering);
	}
};

struct TestSetup {
	std::string file;
	int max_depth;
	int max_num_nodes;

	std::string ToString() const {
		std::ostringstream os;
		os << "File: " << file << ", D=" << max_depth << ", N=" << max_num_nodes;
		return os.str();
	}
};

struct AccuracyTestSetup : TestSetup {
	int correct_solution;
};

struct F1ScoreTestSetup : TestSetup {
	std::vector<F1ScoreSol> correct_solutions;
};

struct RegressionTestSetup : TestSetup {
	double correct_solution;
};

struct SurvivalAnalysisSetup : TestSetup {
	double correct_solution;
};

void RunAccuracyTest(const AccuracyTestSetup& test, const SolverSetup& solver_setup) {
	ParameterHandler parameters = STreeD::ParameterHandler::DefineParameters();
	parameters.SetIntegerParameter("max-depth", test.max_depth);
	parameters.SetIntegerParameter("max-num-nodes", test.max_num_nodes);
	parameters.SetStringParameter("file", test.file);
	solver_setup.Apply(parameters);
	auto rng = std::default_random_engine(0);
	int correct_solution = test.correct_solution;

	std::shared_ptr<SolverResult> result;
	AData data;
	ADataView train_data, test_data;
	try {
		auto solver = new STreeD::Solver<Accuracy>(parameters, &rng);
		FileReader::ReadData<Accuracy>(parameters, data, train_data, test_data, &rng);
		solver->PreprocessData(data, true);
		result = solver->Solve(train_data);
		delete solver;
	} catch (std::exception&) {
		std::ostringstream os;
		os << "An exception occured for " << test.ToString() << std::endl
			<< solver_setup.ToString();
		test_failed(os.str().c_str());
	}
	auto score = std::static_pointer_cast<InternalTrainScore<Accuracy>>(result->scores[0]);
	int misclassifications = score->train_value;

	if (misclassifications != correct_solution) {
		std::ostringstream os;
		os << "Incorrect accuracy solution for " << test.ToString() << std::endl
			<< solver_setup.ToString() << std::endl
			<< "Found " << misclassifications << ", instead of correct solution " << correct_solution << std::endl;
		test_assert_m(misclassifications == correct_solution, os.str().c_str());
	}
	

}


void RunF1ScoreTest(const F1ScoreTestSetup& test, const SolverSetup& solver_setup) {
	ParameterHandler parameters = STreeD::ParameterHandler::DefineParameters();
	parameters.SetIntegerParameter("max-depth", test.max_depth);
	parameters.SetIntegerParameter("max-num-nodes", test.max_num_nodes);
	parameters.SetStringParameter("file", test.file);
	solver_setup.Apply(parameters);
	auto rng = std::default_random_engine(0);
	auto correct_solutions = test.correct_solutions;
	std::sort(correct_solutions.begin(), correct_solutions.end(),
		[](const F1ScoreSol& s1, const F1ScoreSol& s2) { return s1.false_negatives < s2.false_negatives; });

	std::shared_ptr<SolverResult> result;
	AData data;
	ADataView train_data, test_data;
	try {
		auto solver = new STreeD::Solver<F1Score>(parameters, &rng);
		FileReader::ReadData<F1Score>(parameters, data, train_data, test_data, &rng);
		solver->PreprocessData(data, true);
		result = solver->Solve(train_data);
		delete solver;
	} catch (std::exception&) {
		std::ostringstream os;
		os << "An exception occured for " << test.ToString() << std::endl
			<< solver_setup.ToString();
		test_failed(os.str().c_str());
	}

	std::vector<F1ScoreSol> solutions;
	for (auto& s : result->scores) {
		auto s_ = std::static_pointer_cast<InternalTrainScore<F1Score>>(s);
		solutions.push_back(s_->train_value);
	}
	std::sort(solutions.begin(), solutions.end(),
		[](const F1ScoreSol& s1, const F1ScoreSol& s2) { return s1.false_negatives < s2.false_negatives; });
	
	if (solutions.size() != correct_solutions.size()) {
		std::ostringstream os;
		os << "Incorrect F1Score number of solutions for " << test.ToString() << std::endl
			<< solver_setup.ToString() << std::endl
			<< "Found " << solutions.size() << " solutions, instead of " << correct_solutions.size() << " solutions." << std::endl;
		test_assert_m(solutions.size() == correct_solutions.size(), os.str().c_str());
	}

	for (size_t i = 0; i < solutions.size(); i++) {
		if (solutions[i] != correct_solutions[i]) {
			std::ostringstream os;
			os << "Incorrect F1Score solution for " << test.ToString() << std::endl
				<< solver_setup.ToString() << std::endl
				<< "Found solution (" << solutions[i].false_negatives << ", " << solutions[i].false_positives <<"), "
				<< "instead of solution " << correct_solutions[i].false_negatives << ", " << correct_solutions[i].false_positives << ")." << std::endl;
			test_assert_m(solutions[i] == correct_solutions[i], os.str().c_str());
		}
	}


}

void RunRegressionTest(const RegressionTestSetup& test, const SolverSetup& solver_setup) {
	ParameterHandler parameters = STreeD::ParameterHandler::DefineParameters();
	parameters.SetIntegerParameter("max-depth", test.max_depth);
	parameters.SetIntegerParameter("max-num-nodes", test.max_num_nodes);
	parameters.SetStringParameter("file", test.file);
	if (solver_setup.feature_ordering == "gini") return;
	solver_setup.Apply(parameters);
	auto rng = std::default_random_engine(0);
	double correct_solution = test.correct_solution;

	std::shared_ptr<SolverResult> result;
	AData data;
	ADataView train_data, test_data;
	try {
		auto solver = new STreeD::Solver<Regression>(parameters, &rng);
		FileReader::ReadData<Regression>(parameters, data, train_data, test_data, &rng);
		solver->PreprocessData(data, true);
		result = solver->Solve(train_data);
		delete solver;
	} catch (std::exception&) {
		std::ostringstream os;
		os << "An exception occured for " << test.ToString() << std::endl
			<< solver_setup.ToString();
		test_failed(os.str().c_str());
	}
	auto score = std::static_pointer_cast<InternalTrainScore<Regression>>(result->scores[0]);
	double mse = score->score;

	if (std::abs(mse - correct_solution) >= 1e-2) {
		std::ostringstream os;
		os << "Incorrect regression solution for " << test.ToString() << std::endl
			<< solver_setup.ToString() << std::endl
			<< "Found " << mse << ", instead of correct solution " << correct_solution << std::endl;
		test_assert_m(std::abs(mse - correct_solution) < 1e-2 , os.str().c_str());
	}


}

void RunSurvivalAnalysisTest(const SurvivalAnalysisSetup& test, const SolverSetup& solver_setup) {
	ParameterHandler parameters = STreeD::ParameterHandler::DefineParameters();
	parameters.SetIntegerParameter("max-depth", test.max_depth);
	parameters.SetIntegerParameter("max-num-nodes", test.max_num_nodes);
	parameters.SetStringParameter("file", test.file);
	if (solver_setup.feature_ordering == "gini") return;
	solver_setup.Apply(parameters);
	auto rng = std::default_random_engine(0);
	double correct_solution = test.correct_solution;

	std::shared_ptr<SolverResult> result;
	AData data;
	ADataView train_data, test_data;

	FileReader::ReadData<SurvivalAnalysis>(parameters, data, train_data, test_data, &rng);
	try {
		auto solver = new STreeD::Solver<SurvivalAnalysis>(parameters, &rng);
		solver->PreprocessData(data, true);
		result = solver->Solve(train_data);
		delete solver;
	} catch (std::exception&) {
		std::ostringstream os;
		os << "An exception occured for " << test.ToString() << std::endl
			<< solver_setup.ToString();
		test_failed(os.str().c_str());
	}
	auto score = std::static_pointer_cast<InternalTrainScore<SurvivalAnalysis>>(result->scores[0]);
	double error = score->score;

	if (std::abs(error - correct_solution) >= 1e-2) {
		std::ostringstream os;
		os << "Incorrect survival analysis solution for " << test.ToString() << std::endl
			<< solver_setup.ToString() << std::endl
			<< "Found " << error << ", instead of correct solution " << correct_solution << std::endl;
		test_assert_m(std::abs(error - correct_solution) < 1e-2, os.str().c_str());
	}
}


void EnumerateSolverSetupOptions(std::vector<SolverSetup>& solver_setups) {
	std::vector<bool> branch_cache_options{ false, true };
	std::vector<bool> dataset_cache_options{ false, true };
	std::vector<bool> upper_bound_options{ false, true };
	std::vector<bool> lower_bound_options{ false, true };
	std::vector<std::string> feature_selector_options{ "gini", "in-order" };
	int total_options = int(branch_cache_options.size() * dataset_cache_options.size() 
		* upper_bound_options.size() * lower_bound_options.size() * feature_selector_options.size());
	solver_setups.reserve(total_options);

	for (int i = 0; i < total_options; i++) {
		int ix = i;
		SolverSetup s;
		s.use_branch_caching  = ix % 2; ix /= 2;
		s.use_dataset_caching = ix % 2; ix /= 2;
		s.use_upper_bound = ix % 2; ix /= 2;
		s.use_lower_bound = ix % 2; ix /= 2;
		s.feature_ordering = feature_selector_options[ix % 2]; ix /= 2;

		if (!s.use_branch_caching && !s.use_dataset_caching) continue;
		solver_setups.push_back(s);
	}
}

void GetF1ScoreTests(std::vector<F1ScoreTestSetup>& f1_score_tests) {
	f1_score_tests.push_back({
		"data/classification/yeast.csv", 2, 3, { /*										  
			{0, 823}, {1, 784}, {2, 756}, {6, 753}, {9, 733}, {11, 701}, {18, 692}, {21, 679}, {22, 663}, {26, 628}, 
		{33, 605}, {35, 573}, {42, 564}, {45, 551}, {54, 548}, {59, 532}, {64, 521}, {67, 501}, {73, 491}, {77, 463},
		{96, 460}, {97, 456}, {104, 420}, {113, 377}, {139, 372}, {140, 367}, {147, 364}, {154, 358}, {156, 321}, {167, 301},
		{195, 295}, {197, 289}, {202, 258}, {230, 252}, {232, 246}, {235, 230}, {241, 218}, {262, 214}, {263, 196}, {277, 194}, 
		{290, 162}, {312, 159}, {314, 154}, {320, 144}, {323, 132}, {332, 123}, {339, 116}, {350, 97}, {356, 96}, {358, 88}, 
		{363, 76}, {375, 65}, {382, 59}, {390, 54}, {391, 47}, {394, 43}, {404, 38}, {409, 36}, {410, 28}, {417, 25}, {419, 23}, 
		{422, 22}, {423, 17}, {424, 14}, {427, 13}, {430, 12}, {433, 11}, {436, 10}, {438, 8}, {441, 7}, {443, 6}, {447, 5},
		{449, 4}, {451, 3}, {456, 2}, {458, 1}, {461, 0} */
		{77, 463}
		}

	});
}


int main(int argc, char **argv) {
	std::vector<SolverSetup> solver_setups;
	EnumerateSolverSetupOptions(solver_setups);
	

	std::vector<AccuracyTestSetup> accuracy_tests{
		{ "data/fairness/adult-binarized.csv", 2, 3, 8314 },
		{ "data/fairness/adult-binarized.csv", 3, 7, 7514 },
		//{ "data/fairness/adult-binarized.csv", 4, 15, 7188 }
	};
	
	std::vector<F1ScoreTestSetup> f1_score_tests;
	GetF1ScoreTests(f1_score_tests);

	std::vector<RegressionTestSetup> regression_tests{
		{"data/regression/airfoil.csv", 2, 3, 40.0968 },
		{"data/regression/airfoil.csv", 3, 5, 38.26 },
		{"data/regression/airfoil.csv", 4, 15, 33.8628 }
	};

	std::vector<SurvivalAnalysisSetup> survival_analysis_tests{
		{"data/survival-analysis/colon_binary.txt", 2, 3, 0.70269 },
		{"data/survival-analysis/colon_binary.txt", 3, 5, 0.692248 },
		{"data/survival-analysis/colon_binary.txt", 4, 15, 0.652018 }
	};

	for (auto& ss : solver_setups) {
		for (auto& ats : accuracy_tests) {
			RunAccuracyTest(ats, ss);
		}
		for (auto& f1ts : f1_score_tests) {
			RunF1ScoreTest(f1ts, ss);
		}
		for (auto& rts : regression_tests) {
			RunRegressionTest(rts, ss);
		}
		for (auto& sa_ts : survival_analysis_tests) {
			RunSurvivalAnalysisTest(sa_ts, ss);
		}
	}

	
}