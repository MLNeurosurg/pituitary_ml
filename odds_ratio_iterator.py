

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from sklearn.preprocessing import minmax_scale

os.chdir("/Users/toddhollon/Desktop/PituitaryResearch")

# Import raw data
pit_df = pd.read_excel("Pituitary_outcomes_for_python.xlsx", header = 0)

##### Data cleaning and DataFrame building ######
pit_df_filter = pit_df.loc[:,[
    # Demographics
    'TumorType',
    'Macroadenoma',
    'AgeatSurgery',
    'Gender',
    'Race',
    # comorbidity
    "Obesity",
    'BMI',
    'HoMI',
    'HoCHF',
    'HoStroke',
    'Immunesuppression',
    'Hopulmonarydisease',
    'DM',
    'RenalDisease',
    'LiverDisease',
    # Medications
    'Bloodthinners',
    # Surgical Risk
    'Horadiationtoskullbase',
    'Hopriorpituitarysurgery',
    'Preopvisualfieldcut',
    'PreopVisualacuity',
    'PostopNalowest',
    'PostopNahighest',
    # Surgical outcomes
    'DiabetesInsipidus',
    'DesmopressinRequired',
    'Hyponatremia',
    'CranialNerveInjury',
    'CSFLeak',
    "DVTorPE",
    'IntracranialInfection',
    'TensionPneumocephalus',
    'PostopVisualfieldchange',
    'PostopVisualacuitychange']]

# Function to convert age vector
def age_function(x):
    year = x[0:2]
    return(year)
pit_df_filter.AgeatSurgery = pit_df_filter.AgeatSurgery.apply(age_function).astype(float)

# discretize the age variable to greater or less than 40
def age_binary(x):
    if x > 40:
        return 1
    else:
        return 0
pit_df_filter["age_binary"] = pit_df_filter.AgeatSurgery.apply(age_binary)

# Function to define tumor classes
def tumor_rename(x):
    if x[0] == 'a' or x[0] == 'A': # Acromegaly  = 2
        return 2
    elif x[0] == 'C' or x[0] == 'c': # Cushings = 3
        return 3
    elif x[0] == 'p' or x[0] == 'P': # Prolactinoma = 4
        return 4
    elif x[0] == 'n' or x[0] == 'N': # Nonfunctioning = 1
        return 1
    else:
        return 5 # TSHoma = 5
pit_df_filter.TumorType = pit_df_filter.TumorType.apply(tumor_rename)

# Function to revalue sex
def recode_sex(sex_value):
    # Return 1 if sex_value is 'Male'
    if sex_value == 'M':
        return 1
    # Return 0 if sex_value is 'Female'
    elif sex_value == 'F':
        return 0
    # Return np.nan
    else:
        return np.nan
pit_df_filter.Gender = pit_df_filter.Gender.apply(recode_sex)

# Function to revalue race
def recode_race(race):
    if race == 'W':
        return 1
    elif race == 'B':
        return 2
    else:
        return 3
pit_df_filter.Race = pit_df_filter.Race.apply(recode_race)

# Rescale the continous values to be between 0 and 1
col_list = ['BMI', 'AgeatSurgery', 'PostopNalowest', 'PostopNahighest']
for i in col_list:
    pit_df_filter[i] = minmax_scale(pit_df_filter.loc[:,[i]])

#########
# Defining the tumor vectors
pit_df_filter["cushing_vector"] = np.asarray(pit_df_filter.TumorType == 3, dtype=int)
pit_df_filter["acromegaly_vector"] = np.asarray(pit_df_filter.TumorType == 2, dtype=int)
pit_df_filter["nonfunctioning_vector"] = np.asarray(pit_df_filter.TumorType == 1, dtype=int)
pit_df_filter["prolactinoma_vector"] = np.asarray(pit_df_filter.TumorType == 4, dtype=int)

predictor_df = pit_df_filter.loc[:,[
    # Demographics
    "age_binary",
    'Gender',
    'cushing_vector',
    "acromegaly_vector",
    "nonfunctioning_vector",
    "prolactinoma_vector",
    "Macroadenoma",
    "Obesity",
    # 'DM',
    'HoCHF',
    'Hopulmonarydisease',
    'RenalDisease',
    'LiverDisease',
    'Horadiationtoskullbase',
    'Hopriorpituitarysurgery',
    'Bloodthinners']]

def binary_fun(x):
    if x > 0:
        return 1
    else:
        return 0
# vision_pred = pit_df.loc[:,['Preopvisualfieldcut','PreopVisualacuitychange']].sum(axis=1).apply(binary_fun)
# predictor_df["vision_pred"] = vision_pred

outcome_df = pit_df_filter.loc[:,[
    # 'DiabetesInsipidus',
    'DesmopressinRequired',
    'Hyponatremia',
    'CranialNerveInjury',
    'CSFLeak',
    'TensionPneumocephalus',
    'DVTorPE',
    'IntracranialInfection']]
    # 'PostopVisualfieldchange',
    # 'PostopVisualacuitychange']]
target_df = outcome_df.sum(axis = 1)

# Defining the final features and targets
targets = target_df.apply(binary_fun)

##########################################
'''
This is the univariate Odds ratios block for data exploration
'''
import matplotlib
import matplotlib.colors as colors
cmap=matplotlib.cm.RdBu_r # set the colormap to something diverging

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	# http://chris35wills.github.io/matplotlib_diverging_colorbar/

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
mid_val = 0

def odds_ratio_iterator(predictors, outcomes):
    """
    This function will calculate the Odds ratios and corresponding pvalues for qualitative binary outcomes in two dataframes
    :param predictors: Predictors
    :param outcomes: Outcomes
    :return: odds ratio matrix and
    """
    odds_ratios = []
    p_values = []
    for column_pred in predictors:
        for column_out in outcomes:
            cross_tabs = np.asarray(pd.crosstab(predictors[column_pred], outcomes[column_out]))
            odds_ratio, pvalue = fisher_exact(cross_tabs)
            odds_ratios.append(odds_ratio)
            p_values.append(pvalue)
    for i, val in enumerate(odds_ratios):
        if val < 1:
            odds_ratios[i] = -1/val # correct for decimals and "protective" outcomes
    # Each row is a predictor and each column is the outcomes
    odds_matrix = np.asarray(odds_ratios).reshape(predictors.shape[1], outcomes.shape[1])
    pvals_matrix = np.asarray(p_values).reshape(predictors.shape[1], outcomes.shape[1])
    return odds_matrix, pvals_matrix
odds_out, pval_out = odds_ratio_iterator(predictors=predictor_df, outcomes=outcome_df)

def odds_ratio_cleaning(odds_ratios_matrix):
    """
    Function to remove inf, -inf and nan values
    """
    flattened_array = odds_ratios_matrix.flatten()
    flattened_array = flattened_array.round(3)
    for i, val in enumerate(flattened_array):
        if np.any(np.absolute(val) < -0.0001) or np.any(np.absolute(val) > 1000) or np.isnan(val):
            flattened_array[i] = 0
    return flattened_array.reshape(odds_ratios_matrix.shape[0], odds_ratios_matrix.shape[1])
odds_out_cleaned = odds_ratio_cleaning(odds_out)

def Odds_ratio_masking_function(p_values, odds):
    x_unfold = p_values.reshape(p_values.shape[0]*p_values.shape[1],)
    x_list = []
    for i in x_unfold:
        if i >= 0.05:
            x_list.append(0)
            continue
        else:
            x_list.append(i)

    boolean_pvalue_mask = np.ma.make_mask(np.array(x_list).reshape(p_values.shape[0], p_values.shape[1]))
    masked_odds = odds * boolean_pvalue_mask
    return masked_odds, boolean_pvalue_mask
mask_odds, mask_pvalue = Odds_ratio_masking_function(pval_out, odds_out_cleaned)
masked_cleaned = odds_ratio_cleaning(mask_odds)

plt.imshow(masked_cleaned, cmap=cmap, clim=(odds_out_cleaned.min(), odds_out_cleaned.max()), norm=MidpointNormalize(midpoint=mid_val,vmin=odds_out_cleaned.min(), vmax=odds_out_cleaned.max()))
plt.axis("off")
plt.colorbar()


def p_value_function(x):
    x_unfold = x.reshape(x.shape[0]*x.shape[1],)
    x_list = []
    for i in x_unfold:
        if i >= 0.05:
            x_list.append(0)
            continue
        else:
            # x_list.append(np.log(i))
            x_list.append(i)
    return np.array(x_list).reshape(x.shape[0], x.shape[1])
pvalue_out = p_value_function(pval_out)
plt.imshow(pvalue_out, cmap="hot_r")
plt.axis('off')
plt.colorbar()

import seaborn as sns
g = sns.heatmap(masked_cleaned, yticklabels=list(predictor_df.columns.values), xticklabels=list(outcome_df.columns.values), cmap=cmap)
g.set_xticklabels(list(outcome_df.columns.values), rotation=45)
g.set_yticklabels(reversed(list(predictor_df.columns.values)), rotation=45)


########## DEFINING THE TARGET
# Defining the target vector
target_df = pit_df.loc[:,['LengthofStay',
                         'Day30readmission',
                         'Day30EDvisits',
                         'Day30mortality',
                         'RespiratoryFailure',
                         'DVTorPE',
                         'SevereArrhythmia',
                         'Stroke',
                         'MI',
                          'CSFLeak',
                          'TensionPneumocephalus']]
target_array = np.asarray(target_df)

# Function to binarize length of stay
def los_function(x):
    if x > 7:
        return(1)
    else:
        return(0)
target_df.LengthofStay = target_df.LengthofStay.apply(los_function)
sum_targets = target_array.sum(axis=1)
plt.hist(sum_targets, bins=6) # supplemental figure showing how complications cluster

target_df = target_df.sum(axis = 1)
def target_fun(x):
    if x > 0:
        return 1
    else:
        return 0

# Defining the final features and targets
targets = target_df.apply(target_fun)

# remove old targets
features =  pit_df_filter

######## 3d plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(pit_df_filter['AgeatSurgery'], pit_df_filter['PostopNahighest'], pit_df_filter['PostopNalowest'], c=targets, cmap = 'viridis')
plt.show()

# fig = plt.figure()
# ax = Axes3D(fig)
plt.scatter(pit_df_filter['PostopNahighest'], pit_df_filter['PostopNalowest'], c=targets, cmap = 'viridis')
plt.show()

import seaborn as sns
sns.regplot(pit_df_filter['PostopNahighest'], pit_df_filter['PostopNalowest'], fit_reg=False)


from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
# Loading some example data, MUST NOT scale the data
X = np.vstack((pit_df['PostopNalowest'], pit_df['PostopNahighest'])).T.astype(int)
y = np.asanyarray(targets)

# Training a classifier
svm = SVC(C=2, kernel='rbf')
svm_fit = svm.fit(X, y)
svm_fit.predict(X)
from sklearn.metrics import accuracy_score
accuracy_score(y, svm_fit.predict(X))

# Plotting decision regions
plot_decision_regions(np.asanyarray(X), y, clf=svm,
                      res=0.02, legend=2)

# Adding axes annotations
plt.xlabel('Highest postoperative sodium')
plt.ylabel('Lowest postoperative sodium')
plt.title('SVM on Pituitary Dataset')
plt.show()


#### Plotting multiple classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Initializing Classifiers
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(n_estimators=500, random_state=1)
clf3 = GaussianNB()
clf4 = SVC(kernel="poly", degree=3)

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools

gs = gridspec.GridSpec(2, 2)

fig = plt.figure(figsize=(10,8))

labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes (gaussian)', 'Support Vector Machine (3rd degree polynomial kernel)']
for clf, lab, grd in zip([clf1, clf2, clf3, clf4],
                         labels,
                         itertools.product([0, 1], repeat=2)):

    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    plt.title(lab)

plt.show()
