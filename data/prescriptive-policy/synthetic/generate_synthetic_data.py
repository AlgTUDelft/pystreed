import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.preprocessing import KBinsDiscretizer, binarize
import scipy.stats as sps
import os

output_folder = ""

# Also consider other setups as described in https://www.pnas.org/doi/full/10.1073/pnas.1510489113

K = [0,1]
num_features = 20
x_cols = [f"x{f+1}" for f in range(num_features)]

if num_features == 2:
    def phi(*xs):
        return 0.5 * xs[0] + xs[1]
    def kappa(*xs):
        return 0.5 * xs[0]
elif num_features == 10:
    def phi(*xs):
        return 0.5 * sum([xs[i] for i in range(0,2)]) + sum([xs[i] for i in range(2,6)])
    def kappa(*xs):
        return sum([max(0,xs[i]) for i in range(0,2)])
else:
    def phi(*xs):
        return 0.5 * sum([xs[i] for i in range(0,4)]) + sum([xs[i] for i in range(4,8)])
    def kappa(*xs):
        return sum([max(0,xs[i]) for i in range(0,4)])
    

#def phi(x1, x2):
#    return 0.5 * x1 + x2

#def kappa(x1):
#    return 0.5 * x1

def y(k, *xs):
    return phi(*xs) + 0.5*(2*k - 1)* kappa(*xs)

def y_(x,k):
    return np.vectorize(y, excluded="k")(k, *x.T)  

def k(y0, y1, p):
    if random.uniform(0,1) <= p:
        return np.argmax([y0, y1])
    return np.argmin([y0, y1])

def best_k(y0, y1):
    return np.vectorize(lambda y0_, y1_: np.argmax([y0_, y1_]))(y0, y1)

def k_(y0, y1, p):
    return np.vectorize(k, excluded="p")(y0, y1, p)

def combine(y0, y1, k, *xs):
    if k == 0:
        return np.array(list(xs) + [k, y0])
    return np.array(list(xs) + [k, y1])

def combine_(x, y0, y1, k):
    return np.vectorize(combine, signature="(),(),()"+(",()" * num_features) + "->(k)")(y0, y1, k, *x.T)

def dm_fit(regr, df_train, bin_x_train, *args, **kwargs):
    k0_index = df_train["k"] == 0
    k1_index = df_train["k"] == 1
    #Direct method - Linear regression
    reg0 = regr(*args, **kwargs).fit(df_train[k0_index][x_cols], df_train[k0_index]["y"])
    reg1 = regr(*args, **kwargs).fit(df_train[k1_index][x_cols], df_train[k1_index]["y"])
    dm_y0 = pd.Series(reg0.predict(df_train[x_cols]), name="y_k=0")
    dm_y1 = pd.Series(reg1.predict(df_train[x_cols]), name="y_k=1")
    
    # reg0 = regr(*args, **kwargs).fit(bin_x_train[k0_index], df_train[k0_index]["y"])
    # reg1 = regr(*args, **kwargs).fit(bin_x_train[k1_index], df_train[k1_index]["y"])
    # dm_y0 = pd.Series(reg0.predict(bin_x_train), name="y_k=0")
    # dm_y1 = pd.Series(reg1.predict(bin_x_train), name="y_k=1")
    
    dm_train = pd.concat([bin_x_train, dm_y0, dm_y1], axis=1)
    return dm_train 

def ipw_fit(classifier, df_train, bin_x_train, *args, **kwargs):
    ipw = classifier(*args, **kwargs)
    X = df_train[x_cols]
    y = df_train["k"]
    ipw.fit(X, y)   
    probs = pd.DataFrame(ipw.predict_proba(X))
    idx, cols = pd.factorize(df_train['k'])
    probs2 = probs.reindex(cols, axis=1).to_numpy()[np.arange(len(probs)), idx]
    probs3 = pd.Series(probs2, name="mu")
    
    return pd.concat([bin_x_train, df_train[["k", "y"]], probs3], axis=1)

def true_ipw(train_df, bin_x_train, p):
    ipws = pd.Series(train_df.apply(lambda row: p if ((row["x1"] > 0) != (row['k'] == 0)) else 1-p, axis=1), name="mu") 
    return pd.concat([bin_x_train, train_df[["k", "y"]], ipws], axis=1)

def combine_dm_ipw(dm_df, ipw_df):
    return pd.concat([ipw_df, dm_df[["y_k=0", "y_k=1"]]], axis=1)

for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
    for n in range(1, 6):    
        x_train = np.random.normal(0, 1,   (  500, num_features))
        x_test  = np.random.normal(0, 1,   (10000, num_features))
        e_train_0  = np.random.normal(0, 0.1, (  500,  )) # or 0.01
        e_test_0  = np.random.normal(0, 0.1, (10000,  )) # or 0.01
        e_train_1 = e_train_0
        e_test_1 = e_test_0
        #e_train_1 = np.random.normal(0, 0.1, (  500,  )) # or 0.01
        #e_test_1 = np.random.normal(0, 0.1, (10000,  )) # or 0.01
        
        y_train_0 = y_(x_train, 0)# + e_train 
        y_train_1 = y_(x_train, 1)# + e_train
        #y_train = pd.DataFrame({"y_0": y_train_0, "y_1": y_train_1})
        
        y_train_0 += e_train_0
        y_train_1 += e_train_1
        
        y_test_0 = y_(x_test, 0)# + e_test
        y_test_1 = y_(x_test, 1)# + e_test
        #y_test = pd.DataFrame({"y_0": y_test_0, "y_1": y_test_1})
        
        k_train = k_(y_train_0, y_train_1, p)
        
        
        
        y_test_0 += e_test_0
        y_test_1 += e_test_1
        
        k_train_opt = best_k(y_train_0, y_train_1)
        k_test = best_k(y_test_0, y_test_1)
        
        y_train = pd.DataFrame({"y_0": y_train_0, "y_1": y_train_1})
        y_test = pd.DataFrame({"y_0": y_test_0, "y_1": y_test_1})
    
        df_train = pd.DataFrame(combine_(x_train, y_train_0, y_train_1, k_train), columns=x_cols+["k", "y"])
        df_train["k"] = df_train["k"].astype(int)
        
        df_train_opt = pd.DataFrame(combine_(x_train, y_train_0, y_train_1, k_train_opt), columns=x_cols+["k", "y"])
        df_train_opt["k"] = df_train_opt["k"].astype(int)
        
        #df_test =  pd.DataFrame(combine_(x_test, y_test_0, y_test_1, k_test) , columns=x_cols+["k", "y"])
        df_test = pd.concat([pd.DataFrame(x_test, columns=x_cols),
                             pd.Series(k_test, name="k"),
                             pd.Series(y_test_0, name="y_k=0"),
                             pd.Series(y_test_1, name="y_k=1")], axis=1)
        df_test["k"] = df_test["k"].astype(int)
        
        # Binarization
        #binarizer = KBinsDiscretizer(n_bins=10, encode="onehot-dense", strategy="quantile")
        #binarizer.fit(x_train)
        #bin_columns = [f"x1_{i}" for i in range(10)] + [f"x2_{i}" for i in range(10)] 
        #bin_x_train = pd.DataFrame(binarizer.transform(x_train), columns=bin_columns, dtype=int)
        #bin_x_test = pd.DataFrame(binarizer.transform(x_test), columns=bin_columns, dtype=int)
        
        dist = sps.norm(loc=0, scale=1)
        #quantiles = list(np.arange(1.0/11, 1.0, 1.0/11))
        quantiles = list(np.arange(0.1, 1.0, 0.1))
        edges = [dist.ppf(q) for q in quantiles]# + [np.inf] #[-np.inf] +
        
        train_bin_dfs = []
        test_bin_dfs = []
        for fi in range(1,num_features+1):
            labels = [f"x{fi}_{i}" for i in range(10)]
            
            bin_df = pd.DataFrame({f"x{fi}_{i}": binarize(x_train[:, fi-1].reshape(-1, 1), threshold=edges[i]).squeeze() for i,q in enumerate(edges)}, dtype=int)
            #cat_df = pd.cut(x_train[:, fi-1], bins = edges, labels=labels)
            #bin_df = pd.get_dummies(cat_df)
            train_bin_dfs.append(bin_df)
            
            bin_df = pd.DataFrame({f"x{fi}_{i}": binarize(x_test[:, fi-1].reshape(-1, 1), threshold=edges[i]).squeeze() for i,q in enumerate(edges)}, dtype=int)
            #cat_df = pd.cut(x_test[:, fi-1], bins = edges, labels=labels)
            #bin_df = pd.get_dummies(cat_df)
            test_bin_dfs.append(bin_df)
            
        bin_x_train = pd.concat(train_bin_dfs, axis=1)
        bin_x_test = pd.concat(test_bin_dfs, axis=1)
        
         
        #df_train.to_csv(f"datasets/synthetic-factuals-train-p={p}-{n}.csv", sep=',', index=False)
        #df_test.to_csv(f"datasets/synthetic-factuals-test-p={p}-{n}.csv",  sep=',', index=False)
        df_bin_test = pd.concat([bin_x_test, df_test[["k", "y_k=0", "y_k=1"]]], axis=1)
        #df_bin_test.to_csv(f"datasets/synthetic/test-f{num_features}-p={p}-{n}.csv",  sep=',', index=False)
        
        # Direct method - Linear regression
        dm_lr_train = dm_fit(LinearRegression, df_train, bin_x_train)
        #dm_lr_train.to_csv(f"datasets/synthetic/dm_lr_train-f{num_features}-p={p}-{n}.csv",  sep=',', index=False)
        
        # Direct method - Lasso regresion
        dm_lasso_train = dm_fit(Lasso, df_train, bin_x_train, alpha=0.08)
        #dm_lasso_train.to_csv(f"datasets/synthetic/dm_lasso_train-f{num_features}-p={p}-{n}.csv",  sep=',', index=False)
        
        #IPW - Decision Trees
        ipw_dt_train = ipw_fit(DecisionTreeClassifier, df_train, bin_x_train)
        #ipw_dt_train.to_csv(f"datasets/synthetic/ipw_dt_train-f{num_features}-p={p}-{n}.csv",  sep=',', index=False)
        
        #IPW - Log regressor
        ipw_log_train = ipw_fit(LogisticRegression, df_train, bin_x_train)
        #ipw_log_train.to_csv(f"datasets/synthetic/ipw_log_train-f{num_features}-p={p}-{n}.csv",  sep=',', index=False)
        
        #IPW - true ipws
        ipw_true_train = true_ipw(df_train, bin_x_train, p)
        #ipw_true_train.to_csv(f"datasets/synthetic/ipw_true_train-f{num_features}-p={p}-{n}.csv",  sep=',', index=False)
        
        #Double Robust (DR)
        dms = [("lr", dm_lr_train), ("lasso", dm_lasso_train)]
        ipws = [("dt", ipw_dt_train), ("log", ipw_log_train), ("true", ipw_true_train)]
        test_zeros = pd.DataFrame(np.zeros((len(y_test), 5), dtype=int), columns=["k", "y", "mu", "yhat_0", "yhat_1"]) 
        for dm_name, dm_train_df in dms:
            for ipw_name, ipw_train_df in ipws:
                dr_train = combine_dm_ipw(dm_train_df, ipw_train_df)
                #dr_train.to_csv(f"datasets/synthetic/dr_{dm_name}_{ipw_name}_train-f{num_features}-p={p}-{n}.csv",  sep=',', index=False)
                
                dtc_dr_train = pd.concat([df_train["k"], dr_train[dr_train.columns[num_features*len(edges):]], df_train_opt["k"], y_train, bin_x_train], axis=1)
                dtc_dr_train.to_csv(f"{output_folder}train_f{num_features}_{dm_name}_{ipw_name}_p{p}_{n}.csv",  sep=' ', index=False, header=False)
                dtc_dr_test = pd.concat([df_test["k"], test_zeros, df_test["k"], y_test, bin_x_test], axis=1)
                dtc_dr_test.to_csv(f"{output_folder}test_f{num_features}_{dm_name}_{ipw_name}_p{p}_{n}.csv",  sep=' ', index=False, header=False)
        
        #dtc_train_out = pd.concat([df_train["k"], bin_x_train], axis=1)
        #dtc_test_out  = pd.concat([df_test["k"],  bin_x_test],  axis=1)
        #dtc_train_out.to_csv(f"{output_folder}train_f{num_features}_p{p}_{n}.csv",  sep=' ', index=False, header=False)
        #dtc_test_out.to_csv(f"{output_folder}test_f{num_features}_p{p}_{n}.csv",  sep=' ', index=False, header=False)

print("end") 