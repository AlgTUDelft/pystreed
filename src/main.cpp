/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#include "utils/parameter_handler.h"
#include "utils/stopwatch.h"
#include "solver/solver.h"
#include "tasks/tasks.h"

using namespace std;

int main(int argc, char* argv[]) {
	STreeD::ParameterHandler parameters = STreeD::ParameterHandler::DefineParameters();

	if (argc > 1) {
		parameters.ParseCommandLineArguments(argc, argv);
	} else {
		cout << "No parameters specified." << endl << endl;
		parameters.PrintHelpSummary();
		exit(1);
	}

	if (parameters.GetBooleanParameter("verbose")) { parameters.PrintParametersDifferentFromDefault(); }
	std::default_random_engine rng;
	if (parameters.GetIntegerParameter("random-seed") == -1) { 
		rng = std::default_random_engine(int(time(0)));
	} else { 
		rng = std::default_random_engine(int(parameters.GetIntegerParameter("random-seed")));
	}


	parameters.CheckParameters();
	bool verbose = parameters.GetBooleanParameter("verbose");
	
	STreeD::AData data(int(parameters.GetIntegerParameter("max-num-features")));
	STreeD::ADataView train_data, test_data;
	
	// Initialize the solver and the data based on the optimization task at hand
	STreeD::Stopwatch stopwatch;
	stopwatch.Initialise(0);
	
	STreeD::AbstractSolver* solver;
	std::string task = parameters.GetStringParameter("task");
	if (verbose) { std::cout << "Reading data...\n"; }
	if (task == "accuracy") {
		solver = new STreeD::Solver<STreeD::Accuracy>(parameters, &rng);
		STreeD::FileReader::ReadData<STreeD::Accuracy>(parameters, data, train_data, test_data, &rng);
	} else if (task == "cost-complex-accuracy") {
		solver =  new STreeD::Solver<STreeD::CostComplexAccuracy>(parameters, &rng);
		STreeD::FileReader::ReadData<STreeD::CostComplexAccuracy>(parameters, data, train_data, test_data, &rng);
	} else if (task == "balanced-accuracy") {
		solver = new STreeD::Solver<STreeD::BalancedAccuracy>(parameters, &rng);
		STreeD::FileReader::ReadData<STreeD::BalancedAccuracy>(parameters, data, train_data, test_data, &rng);
	} else if (task == "regression") {
		solver =  new STreeD::Solver<STreeD::Regression>(parameters, &rng);
		STreeD::FileReader::ReadData<STreeD::Regression>(parameters, data, train_data, test_data, &rng);
	} else if (task == "cost-complex-regression") {
		solver =  new STreeD::Solver<STreeD::CostComplexRegression>(parameters, &rng);
		STreeD::FileReader::ReadData<STreeD::CostComplexRegression>(parameters, data, train_data, test_data, &rng);
	} else if (task == "piecewise-linear-regression") {
		solver = new STreeD::Solver<STreeD::PieceWiseLinearRegression>(parameters, &rng);
		STreeD::FileReader::ReadData<STreeD::PieceWiseLinearRegression>(parameters, data, train_data, test_data, &rng);
	} else if (task == "simple-linear-regression") {
		solver = new STreeD::Solver<STreeD::SimpleLinearRegression>(parameters, &rng);
		STreeD::FileReader::ReadData<STreeD::SimpleLinearRegression>(parameters, data, train_data, test_data, &rng);
	} else if (task == "cost-sensitive") {
		solver =  new STreeD::Solver<STreeD::CostSensitive>(parameters, &rng);
		STreeD::FileReader::ReadData<STreeD::CostSensitive>(parameters, data, train_data, test_data, &rng);
	} else if (task == "instance-cost-sensitive") {
		solver = new STreeD::Solver<STreeD::InstanceCostSensitive>(parameters, &rng);
		STreeD::FileReader::ReadData<STreeD::InstanceCostSensitive>(parameters, data, train_data, test_data, &rng);
	} else if (task == "f1-score") {
		solver =  new STreeD::Solver<STreeD::F1Score>(parameters, &rng);
		STreeD::FileReader::ReadData<STreeD::F1Score>(parameters, data, train_data, test_data, &rng);
	} else if (task == "group-fairness") {
		solver =  new STreeD::Solver<STreeD::GroupFairness>(parameters, &rng);
		STreeD::FileReader::ReadData<STreeD::GroupFairness>(parameters, data, train_data, test_data, &rng);
	} else if (task == "equality-of-opportunity") {
		solver =  new STreeD::Solver<STreeD::EqOpp>(parameters, &rng);
		STreeD::FileReader::ReadData<STreeD::EqOpp>(parameters, data, train_data, test_data, &rng);
	} else if (task == "prescriptive-policy") {
		solver =  new STreeD::Solver<STreeD::PrescriptivePolicy>(parameters, &rng);
		STreeD::FileReader::ReadData<STreeD::PrescriptivePolicy>(parameters, data, train_data, test_data, &rng);
	} else if (task == "survival-analysis") {
		solver = new STreeD::Solver<STreeD::SurvivalAnalysis>(parameters, &rng);
		STreeD::FileReader::ReadData<STreeD::SurvivalAnalysis>(parameters, data, train_data, test_data, &rng);
	} else {
		std::cout << "Encountered unknown optimization task: " << task << std::endl;
		exit(1);
	}
	clock_t clock_before_solve = clock();
	std::shared_ptr<STreeD::SolverResult> result;
	// Preprocess the data
	solver->PreprocessData(data, true);
	// Solve with hyper-tuning or directly
	if (verbose) { std::cout << "Optimal tree computation started!\n"; }
	if (parameters.GetBooleanParameter("hyper-tune"))
		result = solver->HyperSolve(train_data);
	else 
		result = solver->Solve(train_data);
	solver->InitializeTest(test_data);
	auto test_result = solver->TestPerformance(result, test_data);
	// report results
	std::cout << "TIME: " << stopwatch.TimeElapsedInSeconds() << " seconds\n";
	std::cout << "CLOCKS FOR SOLVE: " << ((double)clock() - (double)clock_before_solve) / CLOCKS_PER_SEC << "\n";


	if (verbose) {
		if (result->NumSolutions() > 0) {
			if (!result->IsProvenOptimal()) {
				std::cout << std::endl << "Warning: No proof of optimality. Results are best solution found before the time-out." << std::endl << std::endl;
			}

			std::cout << "Solutions: " << result->NumSolutions() << " \tD\tN\t\tTrain \t\tTest\t\tAvg. Path length" << std::endl;
			for (int i = 0; i < result->NumSolutions(); i++) {
				auto train_score = result->scores[i];
				auto test_score = test_result->scores[i];
				std::cout << "Solution " << i << ": \t" << std::setw(2) << result->depths[i] << " \t" << result->num_nodes[i] << " \t\t";
				std::cout << std::setprecision (std::numeric_limits<double>::digits10 + 1) << train_score->score << " \t";
				std::cout << std::setprecision (std::numeric_limits<double>::digits10 + 1) << test_score->score << " \t"; 
				std::cout << test_score->average_path_length << std::endl;

				std::cout << "Tree " << i << ": " << result->tree_strings[i] << std::endl;
				
			}
		} else {
			std::cout << std::endl << "No tree found" << std::endl;
		}
	}



	delete solver;

	cout << endl << "STreeD closed successfully!" << endl;
}

