/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#include "utils/parameter_handler.h"

namespace STreeD {

	ParameterHandler ParameterHandler::DefineParameters() {
		ParameterHandler parameters;

		parameters.DefineNewCategory("Main Parameters");
		parameters.DefineNewCategory("Algorithmic Parameters");
		parameters.DefineNewCategory("Objective Parameters");
		parameters.DefineNewCategory("Task-specific Parameters");

		parameters.DefineStringParameter
		(
			"task",
			"Task to optimize.",
			"accuracy",
			"Main Parameters",
			{ "accuracy", "cost-complex-accuracy", "balanced-accuracy",
			"f1-score", "group-fairness", "survival-analysis",
			"regression", "cost-complex-regression", "piecewise-linear-regression",
			"simple-linear-regression", "equality-of-opportunity", "cost-sensitive",
			"prescriptive-policy", "instance-cost-sensitive" }
		);

		parameters.DefineBooleanParameter
		(
			"hyper-tune",
			"Enable/disable hyper-tuning.",
			false,
			"Main Parameters"
		);

		parameters.DefineStringParameter
		(
			"file",
			"Location to the (training) dataset.",
			"", //default value
			"Main Parameters"
		);

		parameters.DefineStringParameter
		(
			"test-file",
			"Location to the test dataset.",
			"", //default value
			"Main Parameters",
			{},
			true
		);

		parameters.DefineFloatParameter
		(
			"time",
			"Maximum runtime given in seconds.",
			3600, //default value
			"Main Parameters",
			0, //min value
			INT32_MAX //max value
		);

		parameters.DefineIntegerParameter
		(
			"max-depth",
			"Maximum allowed depth of the tree, where the depth is defined as the largest number of *decision/feature nodes* from the root to any leaf. Depth greater than four is usually time consuming.",
			3, //default value
			"Main Parameters",
			0, //min value
			20 //max value
		);

		parameters.DefineIntegerParameter
		(
			"max-num-nodes",
			"Maximum number of *decision/feature nodes* allowed. Note that a tree with k feature nodes has k+1 leaf nodes.",
			7, //default value
			"Main Parameters",
			0,
			INT32_MAX
		);

		parameters.DefineIntegerParameter
		(
			"max-num-features",
			"Maximum number of features that are considered from the dataset (in order of appearance).",
			INT32_MAX, // default value,
			"Main Parameters",
			1,
			INT32_MAX
		);

		parameters.DefineIntegerParameter
		(
			"num-instances",
			"Number of instances that are considered from the dataset (in order of appearance).",
			INT32_MAX, // default value,
			"Main Parameters",
			1,
			INT32_MAX
		);

		parameters.DefineIntegerParameter
		(
			"num-extra-cols",
			"Number of extra columns that need to be read after the label and before the binary feature vector.",
			0, // default value,
			"Main Parameters",
			0,
			INT32_MAX
		);

		parameters.DefineBooleanParameter
		(
			"verbose",
			"Determines if the solver should print logging information to the standard output.",
			true,
			"Main Parameters"
		);

		parameters.DefineBooleanParameter
		(
			"all-trees",
			"Instructs the algorithm to compute trees using all allowed combinations of max-depth and max-num-nodes. Used to stress-test the algorithm.",
			false,
			"Main Parameters"
		);

		parameters.DefineFloatParameter
		(
			"train-test-split",
			"The percentage of instances for the test set",
			0.0, //default value
			"Main Parameters",
			0, //min value
			1.0 //max value
		);

		parameters.DefineBooleanParameter
		(
			"stratify",
			"Stratify the train-test split",
			true,
			"Main Parameters"
		);

		parameters.DefineIntegerParameter
		(
			"min-leaf-node-size",
			"The minimum size of leaf nodes",
			1, // default value
			"Main Parameters",
			1, //min value
			INT32_MAX // max value
		);

		parameters.DefineBooleanParameter
		(
			"use-terminal-solver",
			"Use the special solver for trees of depth two.",
			true,
			"Algorithmic Parameters"
		);

		parameters.DefineBooleanParameter
		(
			"use-similarity-lower-bound",
			"Activate similarity-based lower bounding. Disabling this option may be better for some benchmarks, but on most of our tested datasets keeping this on was beneficial.",
			true,
			"Algorithmic Parameters"
		);

		parameters.DefineBooleanParameter
		(
			"use-upper-bound",
			"Use upper bounding. Disabling this option may be better for some benchmarks, specifically when the number of objectives is high.",
			true,
			"Algorithmic Parameters"
		);

		parameters.DefineBooleanParameter
		(
			"use-lower-bound",
			"Use lower bounding. Disabling this option may be better for some benchmarks, specifically when the number of objectives is high.",
			true,
			"Algorithmic Parameters"
		);

		parameters.DefineBooleanParameter
		(
			"use-task-lower-bound",
			"Use task defined lower bounding for tasks that define a custom lower bound. Disabling this option may be better for some benchmarks if the custom bound takes long to compute.",
			true,
			"Algorithmic Parameters"
		);

		parameters.DefineFloatParameter(
			"upper-bound",
			"Search for a tree better than the provided upper bound (numeric).",
			INT32_MAX, // default
			"Algorithmic Parameters",
			0.0, // min
			DBL_MAX // max
		);

		parameters.DefineStringParameter
		(
			"feature-ordering",
			"Feature ordering strategy used to determine the order in which features will be inspected in each node.",
			"in-order", //default value
			"Algorithmic Parameters",
			{ "in-order", "gini", "mse"}
		);

		parameters.DefineIntegerParameter
		(
			"random-seed",
			"Random seed used only if the feature-ordering is set to random. A seed of -1 assings the seed based on the current time.",
			4,
			"Algorithmic Parameters",
			-1,
			INT32_MAX
		);

		parameters.DefineBooleanParameter
		(
			"use-branch-caching",
			"Use branch caching to store computed subtrees.",
			true, //default value
			"Algorithmic Parameters"
		);

		parameters.DefineBooleanParameter
		(
			"use-dataset-caching",
			"Use dataset caching to store computed subtrees. Dataset-caching is more powerful than branch-caching but may required more computational time.",
			false, //default value
			"Algorithmic Parameters"
		);

		parameters.DefineIntegerParameter
		(
			"duplicate-factor",
			"Duplicates the instances the given amount of times. Used for stress-testing the algorithm, not a practical parameter.",
			1,
			"Algorithmic Parameters",
			1,
			INT32_MAX
		);


		parameters.DefineStringParameter
		(
			"cost-file",
			"Location of the file with information about the cost-sensitive classification.",
			"", //default value
			"Task-specific Parameters",
			{},
			true
		);

		parameters.DefineStringParameter
		(
			"ppg-teacher-method",
			"Type of teacher model for prescriptive policy generation.",
			"DM", //default value
			"Task-specific Parameters",
			{"DM", "IPW", "DR"},
			true
		);

		parameters.DefineFloatParameter
		(
			"discrimination-limit",
			"The maximum allowed percentage of discrimination in the training tree",
			1.0, //default value
			"Task-specific Parameters",
			0, //min value
			1.0 //max value
		);

		parameters.DefineFloatParameter
		(
			"cost-complexity",
			"The cost for adding an extra node to the tree. 0.01 means one extra node is only jusitified if it results in at least one percent better training accuracy score.",
			0.00, // default value
			"Objective Parameters",
			0.0, //min value
			1.0 //max value
		);

		parameters.DefineFloatParameter
		(
			"lasso-penalty",
			"The lasso lambda penalty.",
			0.00, // default value
			"Objective Parameters",
			0.0, //min value
			1e12 //max value
		);

		parameters.DefineFloatParameter
		(
			"ridge-penalty",
			"The ridge gamma penalty.",
			0.00, // default value
			"Objective Parameters",
			0.0, //min value
			1e12 //max value
		);

		parameters.DefineStringParameter
		(
			"regression-bound",
			"The type of bound to use, only for cost-complex-regression task.",
			"equivalent", //default value
			"Task-specific Parameters",
			{ "equivalent", "kmeans" },
			true
		);

		return parameters;
	}
}