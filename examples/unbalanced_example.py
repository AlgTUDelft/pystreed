from pystreed import STreeDClassifier, STreeDRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, average_precision_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

THRESHOLD = 0.9
MAX_DEPTH = 4
N_SPLITS = 20
FPR_POINTS = np.linspace(0, 1, 100)
TUNE_CV_SPLITS = 10

# A relatively unbalanced example dataset
df = pd.read_csv("data/classification/soybean.csv", sep=" ", header=None)
X = df[df.columns[1:]].values
y = df[0].values

roc_results = []
results = []

for i in range(N_SPLITS):
    X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=42+i)

    for method in [ "Accuracy", "Regularized Accuracy", "Balanced Accuracy", "F1-Score", "Mean-Squared Error"]:

        print("---------------------------")
        print(f"Split {i+1} - {method}")
        print("---------------------------")

        max_depth = MAX_DEPTH
        if method == "Accuracy":
            model = STreeDClassifier()
        elif method == "Regularized Accuracy":
            model = STreeDClassifier("cost-complex-accuracy")
        elif method == "Balanced Accuracy":
            model = STreeDClassifier("balanced-accuracy")
        elif method == "F1-Score":
            model = STreeDClassifier("f1-score")
            max_depth -= 1 # because f1-score is harder to train
        else:
            model = STreeDRegressor()

        if method == "Mean-Squared Error":
            tune_model = GridSearchCV(model, param_grid={
                    "max_depth": list(range(2, max_depth+1)), 
                    "cost_complexity": [0.05, 0.01, 0.005, 0.0025, 0.001]
                }, scoring="neg_mean_squared_error", verbose=1, cv=TUNE_CV_SPLITS)
        elif method == "Regularized Accuracy":
            tune_model = GridSearchCV(model, param_grid={
                    "max_depth": list(range(2, max_depth+1)), 
                    "cost_complexity": [0.05, 0.01, 0.005, 0.0025, 0.001]
                }, scoring="roc_auc", verbose=1, cv=TUNE_CV_SPLITS)
        else:
            tune_model = GridSearchCV(model, param_grid={
                    "max_depth": list(range(2, max_depth+1)), 
                    "min_leaf_node_size": [1, 2, 5, 10, 20]
                }, scoring="roc_auc", verbose=1, cv=TUNE_CV_SPLITS)

        tune_model.fit(X_train, y_train)
        print(f"Best parameters: {tune_model.best_params_}")
        model = tune_model.best_estimator_
        
        if method == "Mean-Squared Error":
            yhat = model.predict(X_test)
        else:
            yhat = model.predict_proba(X_test)[:, 1]
        y_pred = (yhat >= THRESHOLD).astype(int)

        auc_score = roc_auc_score(y_test, yhat)
        aprc = average_precision_score(y_test, yhat)
        brier = brier_score_loss(y_test, yhat)
        accuracy = accuracy_score(y_test, y_pred)

        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Sensitivity, Specificity, PPV, and NPV calculations
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        percentage_above_threshold = (yhat >= THRESHOLD).mean()

        results.append({
            "Method": method,
            "AUC": auc_score,
            "APRC": aprc,
            "Accuracy": accuracy,
            "Brier score": brier,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "PPV": ppv,
            "NPV": npv,
            "Above threshold": percentage_above_threshold
        })

        fpr, tpr, thresholds = roc_curve(y_test, yhat)
        tpr_interp = np.interp(FPR_POINTS, fpr, tpr)
        tpr_interp[0] = 0.0
        for _fpr, _tpr in zip(FPR_POINTS, tpr_interp):
            roc_results.append({"Method": method, "FPR": _fpr, "TPR": _tpr})

df = pd.DataFrame(results)
pd.options.display.float_format = '{:.3f}'.format
print(df.groupby("Method").mean())

roc_df = pd.DataFrame(roc_results) 

sns.lineplot(roc_df, x="FPR", y="TPR", hue="Method", style="Method")
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--') 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()

