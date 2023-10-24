from sklearn.datasets import load_wine
from pystreed import STreeDClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

raw_data = load_wine(as_frame=True)
data = raw_data["data"]
target = raw_data["target"]

scores = []

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20, random_state=42 + i)
    
    cart_model = DecisionTreeClassifier(max_depth = 3)

    opt_model = STreeDClassifier("accuracy", max_depth = 3,  
                        continuous_binarize_strategy="tree", n_thresholds=5,
                        time_limit=100)

    for n in range(2, 8):
        
        # Set the number of branching nodes for STreeD and fit
        opt_model.set_params(max_num_nodes = n)
        opt_model.fit(X_train, y_train)
        
        # Set the number of leaf nodes (branching nodes + 1) for CART and fit
        cart_model.set_params(max_leaf_nodes = n + 1)
        cart_model.fit(X_train, y_train)

        opt_y_train_pred = opt_model.predict(X_train)
        opt_y_test_pred = opt_model.predict(X_test)
        
        cart_y_train_pred = cart_model.predict(X_train)
        cart_y_test_pred = cart_model.predict(X_test)

        opt_train_score = accuracy_score(y_train, opt_y_train_pred)
        cart_train_score = accuracy_score(y_train, cart_y_train_pred)
        opt_test_score = accuracy_score(y_test, opt_y_test_pred)
        cart_test_score = accuracy_score(y_test, cart_y_test_pred)
        
        scores.append({"Method": "STreeD", "Number of nodes": n, "Accuracy": opt_train_score* 100, "Score": "Train"})
        scores.append({"Method": "STreeD", "Number of nodes": n, "Accuracy": opt_test_score* 100, "Score": "Test"})
        scores.append({"Method": "CART", "Number of nodes": n, "Accuracy": cart_train_score* 100, "Score": "Train"})
        scores.append({"Method": "CART", "Number of nodes": n, "Accuracy": cart_test_score* 100, "Score": "Test"})
        
results = pd.DataFrame(scores)

ax = sns.lineplot(results, x='Number of nodes', y="Accuracy", hue="Method", style='Score')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()

