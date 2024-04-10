https://maxsat-evaluations.github.io/2021/index.html
This dataset contains data about 439 out of 539 instances from the MaxSAT 2021 unweighted complete track.
The excluded instances were removed as no algorithm was able to solve them before timing out.
For each instance 32 features listed below were extracted and binarized according the threshold listed in the info file.

Each row is one instance, the first value is the index of the best performing algorithm, followed by the differences in runtime between between all algorithms and the best and the binary feature values, with a 1 indicating the feature is smaller than the threshold, or 0 otherwise.

Algorithms in order: MaxHS, CASHWMaxSAT, EvalMaxSAT, UWrMaxSAT, Open-WBO-RES-MergeSAT, Open-WBO-RES-Glucose ,Pacose, Exact

FEATURES:
* Size features
	- 0 -> number of variables v
	- 1 -> number of clauses c
	- 2-4 -> ratios v/c (v/c)^2 (v/c  )^3
	- 5-7 -> reciprocals of above
	- 27-31 -> length of clauses, mean, var, min, max, entropy
* Balance features
	- 8-12 -> fraction of positive to total per clause, mean. var, min, max, entropy
	- 13-15 -> fraction of unary, binary, ternary clauses
	- 16-20 -> fraction of positive occurrence for each variable, mean, var, min, max
* Horn features
	- 21 -> Fraction of horn clauses
	- 22-26 -> occurrences in horn clauses per variables mean, var, min, max, entropy

