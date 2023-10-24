import random
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.preprocessing import KBinsDiscretizer, binarize
from sklearn.model_selection import train_test_split
import scipy.stats as sps
import os
import math

# See the supplementary materials at : https://www.nejm.org/doi/suppl/10.1056/NEJMoa0809329/suppl_file/nejm_intwarfpharmacons_753sa1.pdf
# Check the algorithm in section S1e

output_folder = ""
data_file = "warfarin.csv"

df = pd.read_csv(data_file)

# Drop incomplete rows
df = df[~df["Age"].isna()]
df = df[~df["Height (cm)"].isna()] 
#df = df[df["Height (cm)"]!=""] 
df = df[~df["Weight (kg)"].isna()] 
#df = df[~df["Therapeutic Dose of Warfarin"].isna()]
print("rows: ", len(df))

keep_cols = [
    # VKORC1 -- follow imputation rules in Section S4 of the supplementary appendix, and only keep A/G and A/A since they're relevant in the linear function
]

del_cols = [ 'Gender', 'Target INR', 'Estimated Target INR Range Based on Indication',
       'Subject Reached Stable Dose of Warfarin',
       #'Therapeutic Dose of Warfarin',
    "Race (Reported)", 'Ethnicity (Reported)', 'Ethnicity (OMB)',
    'INR on Reported Therapeutic Dose of Warfarin', 'Current Smoker',
    'VKORC1 QC genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T',
       'VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C',
       'VKORC1 QC genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C',
       'VKORC1 QC genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G',
       'VKORC1 QC genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G',
       'VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G',
       'VKORC1 QC genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G',
       'VKORC1 QC genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G',
       'VKORC1 genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C',
       'VKORC1 QC genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C',
       'VKORC1 -1639 consensus',
       'VKORC1 497 consensus', 'VKORC1 1173 consensus',
       'VKORC1 1542 consensus', 'VKORC1 3730 consensus',
       'VKORC1 2255 consensus', 'VKORC1 -4451 consensus',
       'Comorbidities', 'Diabetes',
       'Congestive Heart Failure and/or Cardiomyopathy', 'Valve Replacement',
       'Medications', 'Aspirin', 'Acetaminophen or Paracetamol (Tylenol)',
       'Was Dose of Acetaminophen or Paracetamol (Tylenol) >1300mg/day',
       'Simvastatin (Zocor)', 'Atorvastatin (Lipitor)', 'Fluvastatin (Lescol)',
       'Lovastatin (Mevacor)', 'Pravastatin (Pravachol)',
       'Rosuvastatin (Crestor)', 'Cerivastatin (Baycol)',
       'Sulfonamide Antibiotics', 'Macrolide Antibiotics',
       'Anti-fungal Azoles', 'Herbal Medications, Vitamins, Supplements',
       'Genotyped QC Cyp2C9*2', 'Genotyped QC Cyp2C9*3',
       'Combined QC CYP2C9', 'CYP2C9 consensus',
       "Therapeutic Dose of Warfarin", 
       'Indication for Warfarin Treatment'
]

col_renames = {
    "VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T": "rs9923231",
    "VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G": "rs8050894",
    "VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G": "rs9934438",
    'VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G': "rs2359612",
    'Carbamazepine (Tegretol)': "carbamazepine",
    'Phenytoin (Dilantin)': "phenytoin",
    'Rifampin or Rifampicin': "rifampin/rifampicin",
    'Amiodarone (Cordarone)': "amiodarone",
    'Race (OMB)': "race",
    'Cyp2C9 genotypes': "Cyp2C9",
    "Height (cm)": "height",
    "Weight (kg)": "weight"
}

df.drop(inplace=True, columns = del_cols)
df.rename(columns = col_renames, inplace=True)

def categorical_encoding(column_name):
    series = df[column_name]
    unique = series.unique()
    assert(len(unique) > 1)
    if len(unique) == 2:
        return (series == unique[0]).to_frame("{}={}".format(column_name, unique[0]))
    elif len(unique) >= 20:
        return None
    else:
        values = [series == unique[i] for i in range(len(unique))]
        return pd.DataFrame(zip(*values), index=series.index, columns=["{}={}".format(column_name, unique[i]) for i in range(len(unique))])
    

def categorical_split(category_names, category_values): 
    def _categorical_split_encoding(column_name):
        series = df[column_name]   
        values = [series.isin(vals) for vals in category_values]
        return pd.DataFrame(zip(*values), index=series.index, columns=category_names)
    return _categorical_split_encoding


##### VKORC1 ######
# Should have a categorical output variable for the SNP rs9923231.
# Use decision rules to impute the categorical variable SNP rs9923231
    # If Race is not "Black or African American" or "Missing or Mixed Race"and rs2359612='C/C'
    #     then impute rs9923231='G/G'
    # If Race is not "Black or African American" or "Missing or Mixed Race"and rs2359612='T/T'
    #     then impute rs9923231='A/A'
    # If Race is not "Black or African American" or "Missing or Mixed Race" and rs2359612='C/T'
    #     then impute rs9923231='A/G'
    # If rs9934438='C/C' then impute rs9923231='G/G'
    # If rs9934438='T/T' then impute rs9923231='A/A'
    # If rs9934438='C/T' then impute rs9923231='A/G'
    # If Race is not "Black or African American" or "Missing or Mixed Race" and rs8050894='G/G'
    #     then impute rs9923231='G/G'
    # If Race is not "Black or African American" or "Missing or Mixed Race" and rs8050894='C/C'
    #     then impute rs9923231='A/A'
    # If Race is not "Black or African American" or "Missing or Mixed Race" and rs8050894='C/G'
    #     then impute rs9923231='A/G'
    # Otherwise keep rs9923231 coded as "Missing"
# Translate this categorical variable  into binary variables for rs9923231 = A/G, or A/A, or unknown
def determine_vkorc1(row):
    if not str(row["rs9923231"])=="nan": return row["rs9923231"] 
    if row["race"] != "Black or African American" and row["race"] != "Missing or Mixed Race":
        if row["rs2359612"] == "C/C": return "G/G"
        if row["rs2359612"] == "T/T": return "A/A"
        if row["rs2359612"] == "C/T": return "A/G"
    if row["rs9934438"] == 'C/C': return "G/G"
    if row["rs9934438"] == 'T/T': return "A/A"
    if row["rs9934438"] == 'C/T': return "A/G"
    if row["race"] != "Black or African American" and row["race"] != "Missing or Mixed Race":
        if row["rs8050894"] == "G/G": return "G/G"
        if row["rs8050894"] == "C/C": return "A/A"
        if row["rs8050894"] == "C/G": return "A/G"
    return "Missing"

df['rs9923231'] = df.apply(determine_vkorc1, axis=1)   

df["rs9923231=A/A"] = df.apply(lambda row: row["rs9923231"] == "A/A", axis=1)
df["rs9923231=A/G"] = df.apply(lambda row: row["rs9923231"] == "A/G", axis=1)
df["rs9923231=NA"] = df.apply(lambda row:  row["rs9923231"] == "Missing", axis=1)

df.drop(columns=["rs9923231", "rs8050894", "rs9934438", "rs2359612"], inplace=True)

#### Race ####
# Asian Race = 1, Asian Race = 1 if self-reported race is Asian, otherwise zero
# Black/African American = 1 if self-reported race is Black or African American, otherwise zero
# Missing or Mixed race = 1 if self-reported race is unspecified or mixed, otherwise zero
df["asian"] = df.apply(lambda row: row["race"] == "Asian", axis=1)
df["black/african_american"] = df.apply(lambda row: row["race"] == "Black or African American", axis=1)
df["race_missing/mixed"] = df.apply(lambda row: row["race"] == "Unknown", axis=1)

df.drop(columns=["race"], inplace=True)


#### Cyp2CP ####
# Keep all Cyp2CP genotypes
# binary variables for CYP2C9 *1/*2, CYP2C9 *1/*3, CYP2C9 *2/*2, CYP2C9 *2/*3, CYP2C9 *3/*3, CYP2C9 genotype unknown
for val in ["*1/*2", "*1/*3", "*2/*2", "*2/*3", "*3/*3", "NA"]:
    df[f"CYP2C9={val}"] = df.apply(lambda row: row["Cyp2C9"] == val, axis=1)
df.drop(columns=["Cyp2C9"], inplace=True)

### Amiodarone ###
# replace unknowns with "not used"
df["amiodarone"] = df["amiodarone"].fillna(0)

#### Enzyme inducer #######
# Enzyme inducer status = 1 if patient taking carbamazepine, phenytoin, rifampin, or rifampicin, otherwise zero
def determine_enzyme_inducer(row):
    if row["carbamazepine"] == "1" or row["phenytoin"] == "1" or row["rifampin/rifampicin"] == "1":
        return 1
    return 0
df["enzyme_enducer"] = df.apply(determine_enzyme_inducer, axis=1)
df.drop(columns=["carbamazepine", "phenytoin", "rifampin/rifampicin"], inplace=True)

#### Age ####
# Split into 5 buckets with bins: (0,2], (2,4], (4, 6], (6,7], (7,9]
ages = [
    ("(0-20)", ["10 - 19"]),
    ("(20-40)", ["20 - 29", "30 - 39"]),
    ("(40-60)", ["40 - 49", "50 - 59"]),
    ("(60-70)", ["60 - 69"]),
    ("(70-90)", ["70 - 79", "80 - 89"]), 
     ]

for _range, vals in ages:
    df[f"age{_range}"] = df.apply(lambda row: row["Age"] in vals, axis=1)
df["Age"].replace({"10 - 19": 2, "20 - 29": 3, "30 - 39": 4, "40 - 49": 5,
                    "50 - 59": 6, "60 - 69": 7, "70 - 79": 8, "80 - 89": 9, "90+": 10, "NA": 0}, inplace=True)
#df.drop(columns=["Age"], inplace=True)

#### Height, weight ####
#Split into quintiles, i.e., 4 equally sized buckets
df["height_cat"] = pd.qcut(df["height"], q=5)
df["weight_cat"] = pd.qcut(df["weight"], q=5)

df = df.assign(**pd.get_dummies(df["height_cat"], prefix="height"))
df = df.assign(**pd.get_dummies(df["weight_cat"], prefix="weight_cat"))

df.drop(columns=["height_cat"], inplace=True)
df.drop(columns=["weight_cat"], inplace=True)

def determine_pharmacogenetic_dosing(row, r=0):
    val = (1+random.uniform(-r, r)) * 5.6044
    val -= (1+random.uniform(-r, r)) * 0.2614 * row["Age"]
    val += (1+random.uniform(-r, r)) * 0.0087 * row["height"]
    val += (1+random.uniform(-r, r)) * 0.0128 * row["weight"]
    val -= (1+random.uniform(-r, r)) * 0.8677 * row["rs9923231=A/G"]
    val -= (1+random.uniform(-r, r)) * 1.6974 * row["rs9923231=A/A"]
    val -= (1+random.uniform(-r, r)) * 0.4854 * row["rs9923231=NA"]
    val -= (1+random.uniform(-r, r)) * 0.5211 * row["CYP2C9=*1/*2"]
    val -= (1+random.uniform(-r, r)) * 0.9357 * row["CYP2C9=*1/*3"]
    val -= (1+random.uniform(-r, r)) * 1.0616 * row["CYP2C9=*2/*2"]
    val -= (1+random.uniform(-r, r)) * 1.9206 * row["CYP2C9=*2/*3"]
    val -= (1+random.uniform(-r, r)) * 2.3312 * row["CYP2C9=*3/*3"]
    val -= (1+random.uniform(-r, r)) * 0.2188 * row["CYP2C9=NA"]
    val -= (1+random.uniform(-r, r)) * 0.1092 * row["asian"]
    val -= (1+random.uniform(-r, r)) * 0.2760 * row["black/african_american"]
    val -= (1+random.uniform(-r, r)) * 0.1032 * row["race_missing/mixed"]
    val += (1+random.uniform(-r, r)) * 1.1816 * row["enzyme_enducer"]
    val -= (1+random.uniform(-r, r)) * 0.5503 * row["amiodarone"]
    val += np.random.normal(0, 0.02)
    val *= val
    val /= 7
    if val <= 3: return 0
    if val >= 7: return 2
    return 1

df["k_opt"] = df.apply(determine_pharmacogenetic_dosing, axis=1)

print("Class 0: {}%".format(sum(df["k_opt"] == 0)/len(df)))
print("Class 1: {}%".format(sum(df["k_opt"] == 1)/len(df)))
print("Class 2: {}%".format(sum(df["k_opt"] == 2)/len(df)))

def random_k(row):
    return random.choice([0,1,2])

def dm_fit(regr, k, bin_x_train, *args, **kwargs):
    k0_index = k == 0
    k1_index = k == 1
    k2_index = k == 2
    ix = bin_x_train.index
    #Direct method - Linear regression
    #x_cols = [x for x in bin_x_train.columns if not x in ["Age", "height", "weight", "k", "y", "k_opt"]]
    x_cols = [x for x in bin_x_train.columns if not x in ["k", "y", "k_opt"]]
    x_cols = [x for x in x_cols if not x.startswith("height_") and not x.startswith("weight_") and not x.startswith("age(")]
    reg0 = regr(*args, **kwargs).fit(bin_x_train[k0_index][x_cols], bin_x_train[k0_index]["y"])
    reg1 = regr(*args, **kwargs).fit(bin_x_train[k1_index][x_cols], bin_x_train[k1_index]["y"])
    reg2 = regr(*args, **kwargs).fit(bin_x_train[k2_index][x_cols], bin_x_train[k2_index]["y"])
    dm_y0 = pd.Series(reg0.predict(bin_x_train[x_cols]), index=ix, name="y_k=0", dtype=int)
    dm_y1 = pd.Series(reg1.predict(bin_x_train[x_cols]), index=ix, name="y_k=1", dtype=int)
    dm_y2 = pd.Series(reg2.predict(bin_x_train[x_cols]), index=ix, name="y_k=2", dtype=int)
    
    k0_pred = (dm_y0 > dm_y1) & (dm_y0 > dm_y2)
    k1_pred = (dm_y1 > dm_y0) & (dm_y1 > dm_y2)
    k2_pred = (dm_y2 > dm_y0) & (dm_y2 > dm_y1)
    
    print("  k0 pred: {}%".format(sum(bin_x_train["k_opt"][k0_pred]==0) / len(bin_x_train["k_opt"][k0_pred]) * 100))
    print("  k1 pred: {}%".format(sum(bin_x_train["k_opt"][k1_pred]==1) / len(bin_x_train["k_opt"][k1_pred]) * 100))
    print("  k2 pred: {}%".format(sum(bin_x_train["k_opt"][k2_pred]==2) / len(bin_x_train["k_opt"][k2_pred]) * 100))
    
    # reg0 = regr(*args, **kwargs).fit(bin_x_train[k0_index], df_train[k0_index]["y"])
    # reg1 = regr(*args, **kwargs).fit(bin_x_train[k1_index], df_train[k1_index]["y"])
    # dm_y0 = pd.Series(reg0.predict(bin_x_train), name="y_k=0")
    # dm_y1 = pd.Series(reg1.predict(bin_x_train), name="y_k=1")
    
    dm_train = pd.concat([dm_y0, dm_y1, dm_y2], axis=1)
    return dm_train 

def ipw_fit(classifier, k, bin_train, *args, **kwargs):
    ipw = classifier(*args, **kwargs)
    x_cols = [x for x in bin_train.columns if not x in ["Age", "height", "weight", "k", "y", "k_opt", "y_k=0", "y_k=1", "y_k=2"]]
    X = bin_train[x_cols]
    y = k
    ipw.fit(X, y)   
    probs = pd.DataFrame(ipw.predict_proba(X))
    idx, cols = pd.factorize(k)
    probs2 = probs.reindex(cols, axis=1).to_numpy()[np.arange(len(probs)), idx]
    probs3 = pd.Series(probs2, index = bin_train.index, name="mu")
    return probs3

x_cols = [col for col in df.columns if not col in ["k_opt", "Age", "height", "weight"]]
ipw_cols = ["k", "y", "mu"]
dm_cols = ["y_k=0", "y_k=1", "y_k=2"]

df[x_cols] = df[x_cols].astype(int)

for r in ["random", 0.06, 0.11]:
    for i in range(5): #5
        if r == "random":
            df["k"] = df.apply(random_k, axis=1)
        else:
            df["k"] = df.apply(determine_pharmacogenetic_dosing, axis=1, r=r)
        
        df["y"] = df.apply(lambda row: row["k"] == row["k_opt"], axis=1).astype(int)
        
        
        
        print("{}\t{}".format(r, sum(df["y"]) / len(df)))
        print("  Class 0: {}%".format(sum(df["k"] == 0)/len(df)))
        print("  Class 1: {}%".format(sum(df["k"] == 1)/len(df)))
        print("  Class 2: {}%".format(sum(df["k"] == 2)/len(df)))
        
        #continue
        
        for sp in range(5):#5
            df_train, df_test = train_test_split(df, test_size=0.25)
            df_train = df_train.copy()
            df_test = df_test.copy()
            df_train[["y_k=0", "y_k=1", "y_k=2"]] = dm_fit(RandomForestRegressor, df_train["k"], df_train)
            df_train["mu"] = ipw_fit(DecisionTreeClassifier, df_train["k"], df_train)
        
            df_test["k"] = df_test["k_opt"]
            df_test["y_k=0"] = (df_test["k"] == 0).astype(int)
            df_test["y_k=1"] = (df_test["k"] == 1).astype(int)
            df_test["y_k=2"] = (df_test["k"] == 2).astype(int)
            
            #STreeD output
            df_test[ipw_cols + dm_cols] = np.zeros((len(df_test), 6), dtype=int)
            df_train["cf_y_k=0"] = (df_train["k_opt"] == 0).astype(int)
            df_train["cf_y_k=1"] = (df_train["k_opt"] == 1).astype(int)
            df_train["cf_y_k=2"] = (df_train["k_opt"] == 2).astype(int)
            df_test["cf_y_k=0"] = (df_test["k_opt"] == 0).astype(int)
            df_test["cf_y_k=1"] = (df_test["k_opt"] == 1).astype(int)
            df_test["cf_y_k=2"] = (df_test["k_opt"] == 2).astype(int)
            #df_train[["k"] + x_cols].to_csv(f"{output_folder}train_{r}_{i}_{sp}.csv",  sep=' ', index=False, header=False)
            #df_test[["k"] + x_cols].to_csv(f"{output_folder}test_{r}_{i}_{sp}.csv",  sep=' ', index=False, header=False)
            df_train[["k"] + ipw_cols + dm_cols + ["k_opt", "cf_y_k=0", "cf_y_k=1", "cf_y_k=2"] + x_cols].to_csv(f"{output_folder}train_rf_dt_{r}_{i}_{sp}.csv",  sep=' ', index=False, header=False)
            df_test[["k"] + ipw_cols + dm_cols + ["k_opt", "cf_y_k=0", "cf_y_k=1", "cf_y_k=2"] + x_cols].to_csv(f"{output_folder}test_rf_dt_{r}_{i}_{sp}.csv",  sep=' ', index=False, header=False)
                     
        
print(df.columns)
