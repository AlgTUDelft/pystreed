# Create a new optimization task for STreeD

This tutorial further explains how STreeD works and how you can add a new optimization task to it.
Here we assume that your new task is called `MyNewTask`.

## Create a new task in the C++ code
1. Copy the `include/tasks/accuracy/accuracy.h` into your new `include/tasks/my_new_task.h`.
2. Copy the `src/tasks/accuracy/accuracy.cpp` into your new `src/tasks/my_new_task.cpp`.
3. Add both new files to the `CMakeList.txt` in the header and source files respectively.
4. Change the class name `Accuracy` in both the new header and source file to `MyNewTask`.
5. Add your new task to every .cpp file that implements STreeD's templates. E.g., `src/solver/solver.cpp`, `src/utils/file_reader.cpp`, etc.
6. Add `"my-new-task"` as a valid value for the `"task"` parameter in `src/solver/define_parameters.cpp`.
7. Check for `"my-new-task"` in `src/main.cpp` and create a new `Solver<MyNewTask>` and call `ReadData<MyNewTask>()`.
8. Check if the code compiles and run STreeD with `-task my-new-task` to see if it runs properly. You are now ready to modify the new task as you wish!

## Modify the new task
To modify your new task, you need to understand a few thins about STreeD. This section first explains the class types for your class and the constants that define an optimization task. It then further explains the functions an optimization task is expected to have. Finally, more attention is given to the constants and functions required to implement the special depth-two solver for your task.

Note that many default values for the types, constants and functions are defined `include/tasks/optimization_task.h`

### Types
Since C++ is strongly typed, STreeD uses templates to work with different class types. You can set the types for your class in your `my_new_task.h` header file. 

* `SolType`: The data type of your solution value (e.g., `int` for misclassification score, `double` for MSE)
* `TestSolType`: The data type of your test solution value. This could differ from `SolType`, e.g., when you compute `double` branching costs while training, but want to ignore this and only measure `int` misclassification costs in test evaluation.
* `LabelType`: The label type of an instance. For classification this is `int` and for regression this is `double`.
* `SolLabelType`: The label type assigned to a leaf node. Commonly this type is the same as `LabelType`. 
* `ET`: The type of the Extra Data per instance (beyond feature and label data). E.g., for `PrescriptivePolicy` the extra counter factual data is stored in a `PPGData` class.
* `ContextType`: The type of the context class (i.e., the state of a search node). Default is `BranchContext` that just stores the current branch.

Implicitly defined:
* `SolContainer`: if totally ordered, this is `Node<MyNewTask>`. Otherwise this is `Container<MyNewTask>`.

### Constants
The following constants (or `constexpr`) define your optimization task and its behavior in STreeD.

* `total_order` (`bool`): `True` if the solution values are totally ordered.
* `custom_leaf` (`bool`): `True` if you provide a custom solve leaf node function. Default is to itreate over the set of possible labels. Therefore, a custom leaf node function is required when the label is not discrete.
* `custom_get_label` (`bool`): `True` if you provide a custom function to get the optimal label for a leaf node, e.g., for regression where the optimal label is the mean of the instance labels. Otherwise the label with minimum cost is selected.
* `has_constraint` (`bool`): `True` if the task has a constraint. E.g., a fairness constraint. Note that a minimum leaf node constraint is specified separately through the parameter `"min-leaf-node-size"`.
* `element_additive` (`bool`): `True` if the solution values are element-wise additive. This means STreeD can use its similarity lower bound. `False` disables the similarity lower bound.

Related to branching costs:
* `has_branching_costs` (`bool`): `True` if the optimization task has branching costs.
* `element_branching_costs` (`bool`): `True` if the branching costs depend on individual instances in the data set (not implemented for the depth-two solver).
* `constant_branching_costs` (`bool`): `True` if the branching costs are constant and do not depend on the context or on the data set, e.g., for cost-complexity pruning.

Related to preprocessing:
* `preprocess_data` (`bool`): `True` if the task performs preprocessing on the data (both train and test). This allows modification of instances before computing.
* `preprocess_train_test_data` (`bool`): `True` if the task performs preprocessing on the train or test data. 

Related to task specific optimizations:
* `custom_lower_bound` (`bool`): `True` if the task provides a custom lower bound.
* `custom_similarity_lb` (`bool`): `True` if the task provides a custom similarity lower bound.
* `check_unique` (`bool`): (Only for tasks that are not totally ordered). If `True`, the `Container` class checks for uniqueness of solutions using the solution value's hash and equal functions. Default is `False`. If the task generate many solutions with the same value, setting this to `True` may improve runtime.

Related to hyper-tuning
* `num_tune_phases` (`int`): The number of phases in the hyper tuning. Default is one.

Best and worst solution values or label
* `worst` (`SolType`): The worst solution value possible, e.g., `INT32_MAX`.
* `best` (`SolType`): The best solution value possible, e.g., `0`.
* `worst_label` (`SolLabelType`): The default label for an unitialized node, e.g., `INT32_MAX`.
* `minimum_difference` (`SolType`): The minimum difference between two non equivalent solutions (e.g., 1 for misclassification score). This is used to compute an upper bound from a given solution. Only used for totally ordered tasks.

### Functions

Related to preprocessing
* `void UpdateParameters(const ParameterHandler& parameters)`: inform the task of the (updated) parameters.
* `void InformTrainData(const ADataView&, const DataSummary&)`: informs the task about the training data (before training).
* `void InformTestData(const ADataView&, const DataSummary&)`: informs the task about the test data (before evaluating).
* `void PreprocessData(AData& data, bool train)`: preprocess the data, with `train == True` if this is the training phase. Only if `preprocess_data` is `True`.
* `void PreprocessTrainData(ADataView& train_data)`: preprocess the training data. `PreprocessTestData` is defined similarly. Only if `preprocess_train_test_data` is `True`.

Related to branching:
* `bool MayBranchOnFeature(int feature)`: return `False` is this feature is not available for branching.
* `void GetLeftContext(const ADataView&, const ContextType& context, int feature, ContextType& left_context)`: update the `left_context` from the `context` when branching left on the specified feature.
* `void GetRightContext(const ADataView&, const ContextType& context, int feature, ContextType& right_context)`: update the `right_context` from the `context` when branching right on the specified feature.

Related to branching costs (only if `has_branching_costs` is `True`):
* `SolType GetBranchingCosts(const ADataView&, const ContextType& context, int feature)`: get the branching costs for the data in the given context when branching on feature. Similarly `GetTestBranchingCosts` returns the branching costs when evaluating the test performance.

Related to (optimizing) the leaf nodes:
* `SolType GetLeafCosts(const ADataView& data, const ContextType& context, SolLabelType label)`: return the leaf costs for the given data in the given context for the assigned label.
* `TestSolType GetTestLeafCosts(const ADataView& data, const ContextType& context, SolLabelType label)`: return the test leaf costs for the given data in the given context for the assigned label.
* `LabelType Classify(const AInstance*, SolLabelType label)`: return the label for the given instance if the label of the leaf node where this instance ends in is `label`.
* `SolContainer SolveLeafNode(const ADataView&, const ContextType&)`: returns the optimal solution for the leaf node defined by the given data and context. Only define this if `custom_leaf` is `True`.

For partially ordered tasks (`total_order` is `False`):
* `static bool Dominates(const SolType& s1, const SolType& s2)`: returns true if `s1` dominates `s2`. Similarly `DominatesInv` checks for reverse dominance. `DominatesD0` and `DominatesD0Inv` evaluate dominance in the root node of the search.
* `static double CalcDiff(const SolType& s1, const SolType& s2)`: returns the distance between two solutions. Used for merging 'similar' solutions.
* `static void Merge(const SolType& s1, const SolType& s2, SolType& out)`: Merge two solutions, such that the merged solution dominates both. Similarly `MergeInv` merges such that the merged solution reverse dominates both. This is used for compressing UBs and LBs.
* `RelaxRootSolution(Node<MyNewTask>& sol)`: removes information from solutions in the root node of the search, which is no longer needed to determine dominance.

Related to testing constraint satisfaction (only for when `has_constraint` is `True`):
* `bool SatisfiesConstraint(const Node<MyNewTask>& sol, const ContextType& context)`: returns true if the solution satisfies the constraint.

Related to score and solution values:
* `static SolType Add(const SolType left, const SolType right)`: return left + right. Idem for `TestAdd` which adds values of `TestSolType`, for `Add(const SolType left, const SolType right, SolType& out)` which returns the value in `out`, and for `Subtract` which subtracts the values and returns the value through `out`.
* `static std::string SolToString(SolType val)`: returns `val` as a string. 
* `static std::string ScoreToString(double val)`: returns `val` as a string. Note that the score is different from the solution value. E.g., the solution value is the misclassification score. The score is the accuracy.
* `static bool CompareScore(SolType v1, SolType v2)`: return true if `v1` is better than `v2`.
* `double ComputeTrainScore(SolType test_value)`: return the training score on the training data. Similarly, `ComputeTrainTestScore` computes the training score on the test data. `ComputeTestTestScore` computes the test score on the test data.

Related to the similarity lower bound:
* `SolType GetWorstPerLabel(LabelType label)`: Returns the worst contribution to the solution value a single instnace of the given label.

Related to the custom lower bound (only if `custom_lower_bound` is `True`)
* `SolContainer ComputeLowerBound(const ADataView& data, const Branch& branch, int max_depth, int num_nodes)`: 

Related to the custom similarity lower bound (only if `custom_similarity_lb` is `True`):
* `PairWorstCount<MyNewTask> ComputeSimilarityLowerBound(const ADataView& data_old, const ADataView& data_new)`: returns `PairWorstCount` that has the subtracted LB of `SolType` and the count of the number of differences. 

Related to hyper-tuning:
* `static TuneRunConfiguration GetTuneRunConfiguration(const ParameterHandler& default_config, const ADataView& train_data, int phase)`: Get the tuning configuration from the given default configuration, for the given training data and the given tuning phase.

### Depth-two solver

If you want to use the depth-two solver, some additional types, constants and functions need to be defined.

#### Types
* `SolD2Type`: The type of solutions in the depth-two solver.
* `BranchSolD2Type`: The type of the branching costs in the depth-two solver.

#### Constants
* `use_terminal` (`bool`): `True` if the task implements a depth-two solver.
* `terminal_compute_context` (`bool`): `True` if the context needs to be computed in the depth-two solver, e.g., for checking constraint satisfaction.
* `terminal_filter` (`bool`): `True` if the depth-two solver should filter non feasible solutions and solutions that are dominated by the upper bound. Default is `False`. Set to `True` if you think this will yield a performance increase.

#### Functions
* `void GetInstanceLeafD2Costs(const AInstance* instance, int org_label, int label, SolD2Type& costs, int multiplier)`: Store the costs of this instance with original label `org_label` when it is assigned `label` as its label. Multiplier is either `1` or `-1`. Note that for tasks with a real label, both `org_label` and `label` is always zero.
* `void ComputeD2Costs(const SolD2Type& d2costs, int count, int label, SolType& costs)`: Compute the costs from the depth-two costs, for the given label (always zero for real labels).
* `bool IsD2ZeroCost(const SolD2Type d2costs)`: Return `True` if the given costs are zero.
* `BranchSolD2Type GetBranchingCosts(const ContextType& context, int feature)`: Get the branching costs in the given context (indepenent of the dataset).
* `SolType ComputeD2BranchingCosts(const BranchSolD2Type& d2costs, int count)`: Get the solution value of the branching costs and the number of instances in the leaf node.
* `SolLabelType GetLabel(const SolD2Type& costs, int count)`: Get the label that should be assigned to the leaf node for the given depth-two costs and count.

## Expose your new task in the python binding
After all the function in C++ are defined, you can expose your new task to the python binding. First, adapt the `bindings.cpp`, then update the python files.

### Adapt the `bindings.cpp`:

1. Add `my_new_task` to the `enum` `task_type`.
2. Add `my_new_task` to the `get_task_type_code` function when `task == "my-new-task"`.
3. Add `DefineSolver<MyNewTask>(m, "MyNewTask");`
4. Add a case `my_new_task` to `initialize_streed_solver` and call `Solver<MyNewTask>`.

### Update the python files

If your task is similar to one of the existing python classes (e.g., `STreeDClassifier`), add it to that class.

Otherwise, create a new python class that inherits from `BaseSTreeDSolver` in `pystreed/base.py`.

1. Define constraints on new parameters, using `_parameter_constraints`.
2. Define the `__init__` constructor and provide default parameters. Pass all paramters except newly defined parameters to `BaseSTreeDSolver`.
3. Store new parameters in this class. Override the `_initialize_param_handler` method.

See the other python classes for examples.
