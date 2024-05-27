#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <variant>
#include "utils/parameter_handler.h"
#include "solver/result.h"
#include "solver/solver.h"
#include "model/data.h"

namespace py = pybind11;
using namespace STreeD;

enum task_type {
    accuracy,
    cost_complex_accuracy,
    regression,
    cost_complex_regression,
    spwl_regression,
    pwl_regression,
    cost_sensitive,
    instance_cost_sensitive,
    f1score,
    group_fairness,
    equality_of_opportunity,
    prescriptive_policy,
    survival_analysis
};

task_type get_task_type_code(std::string& task) {
    if (task == "accuracy") return accuracy;
    else if (task == "cost-complex-accuracy") return cost_complex_accuracy;
    else if (task == "regression") return regression;
    else if (task == "cost-complex-regression") return cost_complex_regression;
    else if (task == "simple-linear-regression") return spwl_regression;
    else if (task == "piecewise-linear-regression") return pwl_regression;
    else if (task == "cost-sensitive") return cost_sensitive;
    else if (task == "instance-cost-sensitive") return instance_cost_sensitive;
    else if (task == "f1-score") return f1score;
    else if (task == "group-fairness") return group_fairness;
    else if (task == "equality-of-opportunity") return equality_of_opportunity;
    else if (task == "prescriptive-policy") return prescriptive_policy;
    else if (task == "survival-analysis") return survival_analysis;
    else {
        std::cout << "Encountered unknown optimization task: " << task << std::endl;
        exit(1);
    }
}

template<class LT, class ET>
void NumpyToSTreeDData(const py::array_t<int, py::array::c_style>& _X,
                        const py::array_t<LT, py::array::c_style>& _y,
                        const std::vector<ET>& extra_data,
                        AData& data, ADataView& data_view) {
    const bool regression = std::is_same<LT, double>::value;
    std::vector<const AInstance*> instances;
    auto X = _X.template unchecked<2>();
    auto y = _y.template unchecked<1>();
    const int num_instances = int(X.shape(0));
    const int num_features = int(X.shape(1));
    
    std::vector<std::vector<const AInstance*>> instances_per_label;
    if (regression) instances_per_label.resize(1);
    ET ed;
    std::vector<bool> v(num_features);
    for (py::size_t i = 0; i < num_instances; i++) {
        LT label = y.size() == 0 ? 0 : y(i);
        if(extra_data.size() > 0) {
            ed = extra_data[i];
        }
        for (py::size_t j = 0; j < num_features; j++) {
            v[j] = X(i,j);
        }
        auto instance = new Instance<LT, ET>(int(i), v, label, ed);
        data.AddInstance(instance);
        if (regression) {
            instances_per_label[0].push_back(instance);
        } else {
            int i_label = static_cast<int>(label);
            if (instances_per_label.size() <= i_label) { instances_per_label.resize(i_label + 1); }
            instances_per_label[i_label].push_back(instance);
        } 
    }
    data.SetNumFeatures(num_features);
	data_view = ADataView(&data, instances_per_label, {});
}

std::vector<bool> NumpyRowToBoolVector(const py::array_t<int, py::array::c_style>& _X) {
    auto X = _X.template unchecked<1>();
    std::vector<bool> v(X.shape(0));
    for (py::size_t j = 0; j < X.shape(0); j++) {
        v[j] = X(j);
    }
    return v;
}

template<class OT>
py::class_<Solver<OT>> DefineSolver(py::module& m, const std::string& name) {
    py::class_<Solver<OT>> solver(m, (name + "Solver").c_str());
    
    solver.def("_update_parameters", [](Solver<OT>& _solver, const ParameterHandler& parameters) {
        py::scoped_ostream_redirect stream(std::cout, py::module_::import("sys").attr("stdout"));
        parameters.CheckParameters();
        _solver.UpdateParameters(parameters);
    });

    solver.def("_get_parameters", &Solver<OT>::GetParameters);

    solver.def("_solve", [](Solver<OT>& _solver,
            const py::array_t<int, py::array::c_style>& X,
            const py::array_t<typename OT::LabelType, py::array::c_style>& y,
            const std::vector<typename OT::ET> extra_data) {
        py::scoped_ostream_redirect stream(std::cout, py::module_::import("sys").attr("stdout"));
        AData train_data;
        ADataView train_data_view;
        NumpyToSTreeDData<typename OT::LabelType, typename OT::ET>(X, y, extra_data, train_data, train_data_view);
        _solver.PreprocessData(train_data);
        if(_solver.GetParameters().GetBooleanParameter("hyper-tune")) {
            return _solver.HyperSolve(train_data_view);
        } else {
            return _solver.Solve(train_data_view);
        }
    });

    solver.def("_predict", [](Solver<OT>& _solver,
            std::shared_ptr<SolverResult>& solver_result,
            const py::array_t<int, py::array::c_style>& X,
            const std::vector<typename OT::ET> extra_data) -> py::object {
        py::scoped_ostream_redirect stream(std::cout, py::module_::import("sys").attr("stdout"));
        SolverTaskResult<OT>* result = static_cast<SolverTaskResult<OT>*>(solver_result.get());
        AData test_data;
        ADataView test_data_view;
        py::array_t<typename OT::LabelType, py::array::c_style> y;
        NumpyToSTreeDData<typename OT::LabelType, typename OT::ET>(X, y, extra_data, test_data, test_data_view);
        _solver.PreprocessData(test_data, false);
        auto tree = result->trees[result->best_index];
        std::vector<typename OT::LabelType> predictions = _solver.Predict(tree, test_data_view);
        return py::array_t<typename OT::LabelType, py::array::c_style>(predictions.size(), predictions.data()); 
    });
    
    solver.def("_test_performance", [](Solver<OT>& _solver,
            std::shared_ptr<SolverResult>& solver_result,
            const py::array_t<int, py::array::c_style>& X,
            const py::array_t<typename OT::LabelType, py::array::c_style>& y_true,
            const std::vector<typename OT::ET> extra_data) {
        py::scoped_ostream_redirect stream(std::cout, py::module_::import("sys").attr("stdout"));
        AData test_data;
        ADataView test_data_view;
        NumpyToSTreeDData<typename OT::LabelType, typename OT::ET>(X, y_true, extra_data, test_data, test_data_view);
        _solver.PreprocessData(test_data, false);
        return _solver.TestPerformance(solver_result, test_data_view);
    });

    solver.def("_get_tree", [](Solver<OT>& _solver,
            std::shared_ptr<SolverResult>& solver_result) {
        auto result = static_cast<SolverTaskResult<OT>*>(solver_result.get());
        return result->trees[result->best_index];
    });

    py::class_<Tree<OT>, std::shared_ptr<Tree<OT>>> tree(m, (name + "Tree").c_str());

    tree.def("is_leaf_node", &Tree<OT>::IsLabelNode, "Return true if this node is a leaf node.");
    tree.def("is_branching_node", &Tree<OT>::IsFeatureNode, "Return true if this node is a branching node.");
    tree.def("get_depth", &Tree<OT>::Depth, "Return the depth of the tree.");
    tree.def("get_num_branching_nodes", &Tree<OT>::NumNodes, "Return the number of branching nodes in the tree.");
    tree.def("__str__", &Tree<OT>::ToString);
    tree.def_readonly("left_child", &Tree<OT>::left_child, "Return a reference to the left child node.");
    tree.def_readonly("right_child", &Tree<OT>::right_child, "Return a reference to the right child node.");
    tree.def_readonly("feature", &Tree<OT>::feature, "Get the index of the feature on this branching node.");
    tree.def_readonly("label", &Tree<OT>::label, "Get the label of this leaf node.");

    return solver;
}

void ExposeStringProperty(py::class_<ParameterHandler>& parameter_handler, const std::string& cpp_property_name, const std::string& py_property_name) {
    parameter_handler.def_property(py_property_name.c_str(), 
        [=](const ParameterHandler& p) { return p.GetStringParameter(cpp_property_name); },
        [=](ParameterHandler& p, const std::string& new_value) { return p.SetStringParameter(cpp_property_name, new_value); }); 
}

void ExposeIntegerProperty(py::class_<ParameterHandler>& parameter_handler, const std::string& cpp_property_name, const std::string& py_property_name) {
    parameter_handler.def_property(py_property_name.c_str(), 
        [=](const ParameterHandler& p) { return p.GetIntegerParameter(cpp_property_name); },
        [=](ParameterHandler& p, const int new_value) { return p.SetIntegerParameter(cpp_property_name, new_value); }); 
}

void ExposeBooleanProperty(py::class_<ParameterHandler>& parameter_handler, const std::string& cpp_property_name, const std::string& py_property_name) {
    parameter_handler.def_property(py_property_name.c_str(), 
        [=](const ParameterHandler& p) { return p.GetBooleanParameter(cpp_property_name); },
        [=](ParameterHandler& p, const bool new_value) { return p.SetBooleanParameter(cpp_property_name, new_value); }); 
}

void ExposeFloatProperty(py::class_<ParameterHandler>& parameter_handler, const std::string& cpp_property_name, const std::string& py_property_name) {
    parameter_handler.def_property(py_property_name.c_str(), 
        [=](const ParameterHandler& p) { return p.GetFloatParameter(cpp_property_name); },
        [=](ParameterHandler& p, const double new_value) { return p.SetFloatParameter(cpp_property_name, new_value); }); 
}

PYBIND11_MODULE(cstreed, m) {
    m.doc() = "This is documentation";
    
    /*************************************
          SolverResult
    ************************************/
    py::class_<SolverResult, std::shared_ptr<SolverResult>> solver_result(m, "SolverResult");

    solver_result.def("is_feasible", &SolverResult::IsFeasible);

    solver_result.def("is_optimal", [](const SolverResult &solver_result) {
        py::scoped_ostream_redirect stream(std::cout, py::module_::import("sys").attr("stdout"));
        return solver_result.IsProvenOptimal();
    });

    solver_result.def("score", [](const SolverResult &solver_result) {
        py::scoped_ostream_redirect stream(std::cout, py::module_::import("sys").attr("stdout"));
        return solver_result.scores[solver_result.best_index]->score;
    });

    solver_result.def("tree_depth", [](const SolverResult &solver_result) {
        py::scoped_ostream_redirect stream(std::cout, py::module_::import("sys").attr("stdout"));
        return solver_result.GetBestDepth();
    });

    solver_result.def("tree_nodes", [](const SolverResult &solver_result) {
        py::scoped_ostream_redirect stream(std::cout, py::module_::import("sys").attr("stdout"));
        return solver_result.GetBestNodeCount();
    });


    /*************************************
           ParameterHandler
     ************************************/
    py::class_<ParameterHandler> parameter_handler(m, "ParameterHandler");

    parameter_handler.def(py::init([]() {
        ParameterHandler parameters = ParameterHandler::DefineParameters();
        return new ParameterHandler(parameters);
    }), py::keep_alive<0, 1>());
    parameter_handler.def("check_parameters", &ParameterHandler::CheckParameters);
    ExposeStringProperty(parameter_handler, "task", "optimization_task");
    ExposeBooleanProperty(parameter_handler, "hyper-tune", "hyper_tune");
    ExposeIntegerProperty(parameter_handler, "max-depth", "max_depth");
    ExposeIntegerProperty(parameter_handler, "max-num-nodes", "max_num_nodes");
    ExposeIntegerProperty(parameter_handler, "random-seed", "random_seed");
    ExposeFloatProperty(parameter_handler, "time", "time_limit");
    ExposeFloatProperty(parameter_handler, "cost-complexity", "cost_complexity");
    ExposeStringProperty(parameter_handler, "feature-ordering", "feature_ordering");
    ExposeBooleanProperty(parameter_handler, "verbose", "verbose");
    ExposeFloatProperty(parameter_handler, "train-test-split", "validation_set_fraction");
    ExposeIntegerProperty(parameter_handler, "min-leaf-node-size", "min_leaf_node_size");
    ExposeBooleanProperty(parameter_handler, "use-branch-caching", "use_branch_caching");
    ExposeBooleanProperty(parameter_handler, "use-dataset-caching", "use_dataset_caching");
    ExposeBooleanProperty(parameter_handler, "use-terminal-solver", "use_terminal_solver");
    ExposeBooleanProperty(parameter_handler, "use-similarity-lower-bound", "use_similarity_lower_bound");
    ExposeBooleanProperty(parameter_handler, "use-upper-bound", "use_upper_bound");
    ExposeBooleanProperty(parameter_handler, "use-lower-bound", "use_lower_bound");
    ExposeBooleanProperty(parameter_handler, "use-task-lower-bound", "use_task_lower_bound");
    ExposeFloatProperty(parameter_handler, "upper-bound", "upper_bound");
    ExposeStringProperty(parameter_handler, "ppg-teacher-method", "ppg_teacher_method");
    ExposeFloatProperty(parameter_handler, "lasso-penalty", "lasso_penalty");
    ExposeFloatProperty(parameter_handler, "ridge-penalty", "ridge_penalty");
    ExposeStringProperty(parameter_handler, "regression-bound", "regression_lower_bound");
    ExposeFloatProperty(parameter_handler, "discrimination-limit", "discrimination_limit");
    
    /*************************************
           Solver
     ************************************/

    DefineSolver<Accuracy>(m, "Accuracy");
    DefineSolver<CostComplexAccuracy>(m, "CostComplexAccuracy");
    DefineSolver<F1Score>(m, "F1Score");
    DefineSolver<Regression>(m, "Regression");
    DefineSolver<CostComplexRegression>(m, "CostComplexRegression");
    DefineSolver<SimpleLinearRegression>(m, "SimpleLinearRegression");
    DefineSolver<PieceWiseLinearRegression>(m, "PieceWiseLinearRegression");
    DefineSolver<SurvivalAnalysis>(m, "Survival");
    DefineSolver<PrescriptivePolicy>(m, "PrescriptivePolicy");
    DefineSolver<GroupFairness>(m, "GroupFairness");
    DefineSolver<EqOpp>(m, "EqOpp");
    DefineSolver<InstanceCostSensitive>(m, "InstanceCostSensitive");
    py::class_<Solver<CostSensitive>> cost_sensitive_solver = DefineSolver<CostSensitive>(m, "CostSensitive");
    cost_sensitive_solver.def("specify_costs", [](Solver<CostSensitive>& solver, const CostSpecifier& cost_specifier) {
        solver.GetTask()->UpdateCostSpecifier(cost_specifier);
    });

    m.def("initialize_streed_solver", [](ParameterHandler& parameters) {
        py::scoped_ostream_redirect stream(std::cout, py::module_::import("sys").attr("stdout"));

        // Create random engine
        std::default_random_engine rng;
	    if (parameters.GetIntegerParameter("random-seed") == -1) { 
		    rng = std::default_random_engine(int(time(0)));
	    } else { 
		    rng = std::default_random_engine(int(parameters.GetIntegerParameter("random-seed")));
	    }

        parameters.CheckParameters();
	    bool verbose = parameters.GetBooleanParameter("verbose");

        STreeD::AbstractSolver* solver;
        std::string task = parameters.GetStringParameter("task");
        switch(get_task_type_code(task)) {
            case accuracy: solver = new Solver<Accuracy>(parameters, &rng); break;
            case cost_complex_accuracy: solver = new Solver<CostComplexAccuracy>(parameters, &rng); break;
            case regression: solver = new Solver<Regression>(parameters, &rng); break;
            case cost_complex_regression: solver = new Solver<CostComplexRegression>(parameters, &rng); break;
            case spwl_regression: solver = new Solver<SimpleLinearRegression>(parameters, &rng); break;
            case pwl_regression: solver = new Solver<PieceWiseLinearRegression>(parameters, &rng); break;
            case cost_sensitive: solver = new Solver<CostSensitive>(parameters, &rng); break;
            case instance_cost_sensitive: solver = new Solver<InstanceCostSensitive>(parameters, &rng); break;
            case f1score: solver = new Solver<F1Score>(parameters, &rng); break;
            case group_fairness: solver = new Solver<GroupFairness>(parameters, &rng); break;
            case equality_of_opportunity: solver = new Solver<EqOpp>(parameters, &rng); break;
            case prescriptive_policy: solver = new Solver<PrescriptivePolicy>(parameters, &rng); break;
            case survival_analysis: solver = new Solver<SurvivalAnalysis>(parameters, &rng); break;
        }
        return solver;
    }, py::keep_alive<0, 1>());


    
    /*************************************
           Extra Data
     ************************************/

    py::class_<SAData>(m, "SAData")
        .def(py::init<int, double>())
        .def_property_readonly("event", &SAData::GetEvent)
        .def_property_readonly("hazard", &SAData::GetHazard);

    py::class_<PPGData>(m, "PPGData")
        .def(py::init<int, double, double, std::vector<double>&, int, std::vector<double>&>())
        .def(py::init<int, double, double, std::vector<double>&>())
        .def(py::init<>())
        .def_readonly("historic_treatment", &PPGData::k)
        .def_readonly("historic_outcome", &PPGData::y)
        .def_readonly("propensity_score", &PPGData::mu)
        .def_readonly("predicted_outcome", &PPGData::yhat)
        .def_readonly("optimal_treatment", &PPGData::k_opt)
        .def_readonly("counterfactual_outcome", &PPGData::cf_y);

    py::class_<InstanceCostSensitiveData>(m, "CostVector")
        .def(py::init<std::vector<double>&>())
        .def(py::init<>())
        .def_readonly("costs", &InstanceCostSensitiveData::costs);

    py::class_<PieceWiseLinearRegExtraData>(m, "ContinuousFeatureData")
        .def(py::init<const std::vector<double>&>())
        .def_readonly("feature_data", &PieceWiseLinearRegExtraData::x);

    py::class_<SimpleLinRegExtraData>(m, "SimpleContinuousFeatureData")
        .def(py::init<const std::vector<double>&>())
        .def_readonly("feature_data", &PieceWiseLinearRegExtraData::x);

        
    /*************************************
           Label
     ************************************/
    py::class_<FeatureCostSpecifier>(m, "FeatureCostSpecifier")
        .def(py::init<double, double, const std::string&, int, int>())
        .def_readonly("feature_cost", &FeatureCostSpecifier::feature_cost)
        .def_readonly("discount_cost", &FeatureCostSpecifier::discount_cost)
        .def_readonly("group_name", &FeatureCostSpecifier::group_name)
        .def_readonly("binary_begin", &FeatureCostSpecifier::binary_begin)
        .def_readonly("binary_end", &FeatureCostSpecifier::binary_end)
        .def("__str__", &FeatureCostSpecifier::ToString);
    
    py::class_<CostSpecifier>(m, "CostSpecifier")
        .def(py::init<const std::string&, int>())
        .def(py::init<const std::vector<std::vector<double>>&, const std::vector<FeatureCostSpecifier>&>());
    
    /*************************************
           Label
     ************************************/

    py::class_<LinearModel>(m, "LinearModel")
        .def_readonly("coefficients", &LinearModel::b)
        .def_readonly("intercept", &LinearModel::b0)
        .def("__str__", &LinearModel::ToString)
        .def("__call__", [](const LinearModel& model, const py::array_t<int, py::array::c_style>& _X, const PieceWiseLinearRegExtraData& extra_data) -> double {
            const auto fv = NumpyRowToBoolVector(_X);
            Instance<double, PieceWiseLinearRegExtraData> instance(0, fv, 0, extra_data);
            return model.Predict(&instance);
        });
}