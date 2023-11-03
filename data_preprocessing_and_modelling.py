#!/usr/bin/env python
# coding: utf-8

# # Importing and Preparation

# In[1]:


# load data
import pandas as pd
import numpy as np

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# modelling
import statsmodels.formula.api as smf
import sklearn as sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score as acc
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing


# In[2]:


from ipynb.fs.full.data_preparation import application


# In[3]:


data = application.copy()
data.head()


# In[4]:


data.info()


# In[5]:


data['DAYS_BIRTH'] = data['DAYS_BIRTH'] * -1
data['DAYS_EMPLOYED'] = data['DAYS_EMPLOYED'] * -1


# In[6]:


# Define response variable
response = 'status'


# In[7]:


y = data[response]
x = data.drop(response, axis=1)


# In[8]:


# Splitting data into train and test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123, stratify = y)


# In[9]:


# Concatenate x_train and y_train as data_train
data_train = pd.concat((x_train, y_train),
                       axis = 1)


# In[10]:


data_train.info()


# # Train Data
# 
# First of all, we need to dive deep into each predictor variables by doing Exploratory Data Analysis (EDAs) to know if there is anything need to be done regarding preprocessing process. In general, EDA here consists of:
# 
# 1. Descriptive statistics (for numerical) or proportion (for categorical)
# 2. Histogram (for numerical) or odds (for categorical)
# 3. Grouping based on target variable
# 4. Boxplot to see outliers
# 5. Imputation for empty cells

# In[11]:


# Create function to do EDA for categorical predictors

def cat_EDA(data, predictor):
    
    # Calculate proportion
    proportion = data[predictor].value_counts(normalize=True)
    
    # Create a crosstab to show proportion of predictor variables based on response variable
    crosstab = pd.crosstab(data[predictor],
                         data[response],
                         margins = False)
    
    # Calculate ods
    crosstab['Odds'] = np.round((crosstab[1]/crosstab[0]),2)
    
    # Sort by odds value
    crosstab = crosstab.sort_values(by = 'Odds',
                       ascending = False)
    
    bar_chart = sns.countplot(data = data, x = predictor, hue = data[response])
    
    return proportion, crosstab, bar_chart


# In[12]:


# Create function to do EDA for categorical predictors (descriptive statistics)

def num_EDA_desc(data, predictor):
    
    # Descriptive statistics
    desc_stat = data[predictor].describe()
    
    # Descriptive statistics based on response
    desc_stat_resp = data[predictor].groupby(data[response]).describe()
    
    return desc_stat, desc_stat_resp


# In[13]:


# Create function to do EDA for categorical predictors (histogram and box plot)

def num_EDA_graph(data, predictor):
    # Creating a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
    f, (ax_hist, ax_box) = plt.subplots(2, 1, figsize=(8, 7))

    # Assigning a graph to each ax
    sns.histplot(data = data, x = predictor, ax = ax_hist)
    sns.boxplot(data = data, x = response, y = predictor, ax = ax_box)

    plt.show()


# ## Code Gender

# In[14]:


proportion_code_gender, crosstab_code_gender, fig_code_gender = cat_EDA(data_train, 'CODE_GENDER')


# In[15]:


# Proportion

proportion_code_gender


# In[16]:


# Calculate odds

crosstab_code_gender


# ## Own Car

# In[17]:


data_train['FLAG_OWN_CAR'].value_counts()


# In[18]:


# Convert value into 1 and 0

data_train.loc[data_train['FLAG_OWN_CAR'] == "Y", 'FLAG_OWN_CAR'] = 1
data_train.loc[data_train['FLAG_OWN_CAR'] == "N", 'FLAG_OWN_CAR'] = 0


# In[19]:


data_train['FLAG_OWN_CAR'].value_counts()


# In[20]:


proportion_own_car, crosstab_own_car, fig_own_car = cat_EDA(data_train, 'FLAG_OWN_CAR')


# In[21]:


# Proportion

proportion_own_car


# In[22]:


# Calculate odds

crosstab_own_car


# ## Own Realty

# In[23]:


data_train['FLAG_OWN_REALTY'].value_counts()


# In[24]:


# Convert value into 1 and 0

data_train.loc[data_train['FLAG_OWN_REALTY'] == "Y", 'FLAG_OWN_REALTY'] = 1
data_train.loc[data_train['FLAG_OWN_REALTY'] == "N", 'FLAG_OWN_REALTY'] = 0


# In[25]:


data_train['FLAG_OWN_REALTY'].value_counts()


# In[26]:


proportion_own_realty, crosstab_own_realty, fig_own_realty = cat_EDA(data_train, 'FLAG_OWN_REALTY')


# In[27]:


# Proportion

proportion_own_realty


# In[28]:


# Calculate odds

crosstab_own_realty


# ## Count Children

# In[29]:


# Descriptive statistics

desc_stat_children, desc_stat_resp_childern = num_EDA_desc(data = data_train, predictor = 'CNT_CHILDREN')


# In[30]:


desc_stat_children


# In[31]:


desc_stat_resp_childern


# In[32]:


# Graph

num_EDA_graph(data = data_train, predictor = 'CNT_CHILDREN')


# ## Annual Income Total

# In[33]:


# Descriptive statistics

desc_stat_annual_inc, desc_stat_resp_annual_inc = num_EDA_desc(data = data_train, predictor = 'AMT_INCOME_TOTAL')


# In[34]:


desc_stat_annual_inc


# In[35]:


desc_stat_resp_annual_inc


# In[36]:


# Graph

num_EDA_graph(data = data_train, predictor = 'AMT_INCOME_TOTAL')


# ## Income Type

# In[37]:


proportion_income_type, crosstab_income_type, fig_income_type = cat_EDA(data_train, 'NAME_INCOME_TYPE')


# In[38]:


# Proportion

proportion_income_type


# In[39]:


# Calculate odds

crosstab_income_type


# ## Education Type

# In[40]:


proportion_education_type, crosstab_education_type, fig_education_type = cat_EDA(data_train, 'NAME_EDUCATION_TYPE')


# In[41]:


# Proportion

proportion_education_type


# In[42]:


# Calculate odds

crosstab_education_type


# ## Family Status

# In[43]:


proportion_famiy_status, crosstab_famiy_status, fig_famiy_status = cat_EDA(data_train, 'NAME_FAMILY_STATUS')


# In[44]:


# Proportion

proportion_famiy_status


# In[45]:


# Calculate odds

crosstab_famiy_status


# ## Housing Type

# In[46]:


proportion_housing_type, crosstab_housing_type, fig_housing_type = cat_EDA(data_train, 'NAME_HOUSING_TYPE')


# In[47]:


# Proportion

proportion_housing_type


# In[48]:


# Odds

crosstab_housing_type


# ## Days Birth

# In[49]:


# Descriptive statistics

desc_stat_days_birth, desc_stat_resp_days_birth = num_EDA_desc(data = data_train, predictor = 'DAYS_BIRTH')


# In[50]:


desc_stat_days_birth


# In[51]:


desc_stat_resp_days_birth


# In[52]:


# Graph

num_EDA_graph(data = data_train, predictor = 'DAYS_BIRTH')


# In[53]:


# Generate value in age

data_train['DAYS_BIRTH'] = data_train['DAYS_BIRTH']/365


# In[54]:


data_train.head()


# In[55]:


# Descriptive statistics

desc_stat_days_birth, desc_stat_resp_days_birth = num_EDA_desc(data = data_train, predictor = 'DAYS_BIRTH')


# In[56]:


desc_stat_days_birth


# In[57]:


desc_stat_resp_days_birth


# In[58]:


# Graph

num_EDA_graph(data = data_train, predictor = 'DAYS_BIRTH')


# In[59]:


data_train.rename(columns={'DAYS_BIRTH':'AGE'}, inplace=True)


# ## Days Employed

# In[60]:


# Descriptive statistics

desc_stat_days_employed, desc_stat_resp_days_employed = num_EDA_desc(data = data_train, predictor = 'DAYS_EMPLOYED')


# In[61]:


desc_stat_days_employed


# In[62]:


desc_stat_resp_days_employed


# In[63]:


# Graph

num_EDA_graph(data = data_train, predictor = 'DAYS_EMPLOYED')


# In[64]:


data_train[data_train['DAYS_EMPLOYED']<1]


# In[65]:


data_train[data_train['DAYS_EMPLOYED']<1].info()


# There are negative values. Needs further look.

# In[66]:


data_train[data_train['DAYS_EMPLOYED']<1]['DAYS_EMPLOYED'].value_counts()


# In[67]:


len(data_train['DAYS_EMPLOYED'])


# In[68]:


data_train[data_train['DAYS_EMPLOYED']<1]['DAYS_EMPLOYED'].value_counts()/len(data_train['DAYS_EMPLOYED'])


# In[69]:


data_train[data_train['DAYS_EMPLOYED']<1]['NAME_INCOME_TYPE'].value_counts()


# Entries with negative values are entries with NAME_INCOME_TYPE 'Pensioner'

# In[70]:


data_train['OCCUPATION_TYPE'].value_counts()


# In[71]:


# Handle errors
data_train.loc[data_train['DAYS_EMPLOYED'] < 1, 'OCCUPATION_TYPE'] = "Pensioner" # There entries are those with 'Pensioner' on NAME_INCOME_TYPE
data_train.loc[data_train['DAYS_EMPLOYED'] < 1, 'DAYS_EMPLOYED'] = np.nan # Impute empty


# In[72]:


# Check missing values
data_train['DAYS_EMPLOYED'].isna().sum()


# In[73]:


data_train['DAYS_EMPLOYED'] = data_train['DAYS_EMPLOYED']/365


# In[74]:


data_train.rename(columns={'DAYS_EMPLOYED':'WORKING_YEARS'}, inplace=True)


# ## Days Birth and Employed Analysis

# In[75]:


data_train.head()


# In[76]:


data_train['AGE'].describe()


# In[77]:


data_train['WORKING_YEARS'].describe()


# In[78]:


data_train[data_train['AGE']<data_train['WORKING_YEARS']]


# In[79]:


data_train[data_train['AGE']<data_train['WORKING_YEARS']]['WORKING_YEARS'].value_counts()


# In[80]:


data_train[data_train['AGE']<data_train['WORKING_YEARS']]['NAME_INCOME_TYPE'].value_counts()


# ## Mobil

# In[81]:


proportion_mobil, crosstab_mobil, fig_mobil = cat_EDA(data_train, 'FLAG_MOBIL')


# In[82]:


# Proportion

proportion_mobil


# In[83]:


# Odds

crosstab_mobil


# ## Work Phone

# In[84]:


proportion_work_phone, crosstab_work_phone, fig_work_phone = cat_EDA(data_train, 'FLAG_WORK_PHONE')


# In[85]:


# Proportion

proportion_work_phone


# In[86]:


# Odds

crosstab_work_phone


# ## Phone

# In[87]:


proportion_phone, crosstab_phone, fig_phone = cat_EDA(data_train, 'FLAG_PHONE')


# In[88]:


# Proportion

proportion_phone


# In[89]:


# Odds

crosstab_phone


# ## Email

# In[90]:


proportion_email, crosstab_email, fig_email = cat_EDA(data_train, 'FLAG_EMAIL')


# In[91]:


# Proportion

proportion_email


# In[92]:


# Odds

proportion_email


# ##  Occupation Type

# In[93]:


data_train['OCCUPATION_TYPE'].isna().sum()


# In[94]:


data_train[data_train['OCCUPATION_TYPE'].isna()]


# In[95]:


data_train['OCCUPATION_TYPE'] = data_train['OCCUPATION_TYPE'].fillna('Unknown')


# In[96]:


data_train['OCCUPATION_TYPE'].isna().sum()


# In[97]:


proportion_occupation_type, crosstab_occupation_type, fig_occupation_type = cat_EDA(data_train, 'OCCUPATION_TYPE')


# In[98]:


# Proportion

proportion_occupation_type


# In[99]:


# Odds

crosstab_occupation_type


# ## Count Family Members

# In[100]:


# Descriptive statistics

desc_stat_family_members, desc_stat_resp_family_members = num_EDA_desc(data = data_train, predictor = 'CNT_FAM_MEMBERS')


# In[101]:


desc_stat_family_members


# In[102]:


desc_stat_resp_family_members


# In[103]:


# Graph

num_EDA_graph(data = data_train, predictor = 'CNT_FAM_MEMBERS')


# ## End of EDA

# In[104]:


data_train.info()


# In[105]:


# Change all flag-related columns into object type

data_train = data_train.astype({'FLAG_MOBIL':object, 'FLAG_WORK_PHONE':object, 'FLAG_PHONE':object, 'FLAG_EMAIL':object, 'FLAG_EMAIL':object})


# In[106]:


data_train.info()


# ## Binning

# In order to simplify our dataset and, more importantly, scaling all numerical variables into same scale with same weight, we do binning.

# In[107]:


# Define categorical predictor

cat_var = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
          'NAME_HOUSING_TYPE', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE']


# In[108]:


# Define numerical predictors

num_var = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AGE', 'WORKING_YEARS', 'CNT_FAM_MEMBERS']


# In[109]:


data_train['CNT_CHILDREN'].value_counts()


# In[110]:


data_train['AMT_INCOME_TOTAL'].value_counts()


# In[111]:


data_train['AGE'].value_counts()


# In[112]:


data_train['WORKING_YEARS'].value_counts()


# In[113]:


data_train['CNT_FAM_MEMBERS'].value_counts()


# In[114]:


# Create a function for binning the numerical predictor
def create_binning(data, predictor_label, num_of_bins):
    """
    Function for binning numerical predictor.

    Parameters
    ----------
    data : array like
      The name of dataset.

    predictor_label : object
      The label of predictor variable.

    num_of_bins : integer
      The number of bins.


    Return
    ------
    data : array like
      The name of transformed dataset.

    """
    # Create a new column containing the binned predictor
    data[predictor_label + "_bin"] = pd.qcut(data[predictor_label],
                                             q = num_of_bins,
                                            duplicates='drop')

    return data


# In[115]:


for column in num_var:
    data_train_binned = create_binning(data = data_train,
                                     predictor_label = column,
                                     num_of_bins = 4)


# In[116]:


data_train_binned.T


# In[117]:


# Check for missing values

data_train_binned.isna().sum()


# In[118]:


# Turning empty values into a distinct category called 'Missing'

data_train_binned['WORKING_YEARS_bin'] = data_train_binned['WORKING_YEARS_bin'].cat.add_categories('Missing')
data_train_binned['WORKING_YEARS_bin'].fillna(value = 'Missing', inplace = True)


# In[119]:


# Check for missing values

data_train_binned.isna().sum()


# In[120]:


# Define the initial empty list
crosstab_num = []

for column in num_var:

  # Create a contingency table
  crosstab = pd.crosstab(data_train_binned[column + "_bin"],
                         data_train_binned[response],
                         margins = True)

  # Append to the list
  crosstab_num.append(crosstab)


# In[121]:


# Define the initial empty list
crosstab_cat = []

for column in cat_var:

  # Create a contingency table
  crosstab = pd.crosstab(data_train_binned[column],
                         data_train_binned[response],
                         margins = True)

  # Append to the list
  crosstab_cat.append(crosstab)


# In[122]:


# Put all two in a crosstab_list
crosstab_list = crosstab_num + crosstab_cat

crosstab_list


# ## Weight of Evidence (WOE) and Information Value (IV)

# In[123]:


# Define the initial list for WOE
WOE_list = []

# Define the initial list for IV
IV_list = []

# Create the initial table for IV
IV_table = pd.DataFrame({'Characteristic': [],
                         'Information Value' : []})

# Perform the algorithm for all crosstab
for crosstab in crosstab_list:

    # Calculate % Good
    crosstab['p_good'] = crosstab[0]/crosstab[0]['All']

    # Calculate % Bad
    crosstab['p_bad'] = crosstab[1]/crosstab[1]['All']

    # Calculate the WOE
    crosstab['WOE'] = np.log(crosstab['p_good']/crosstab['p_bad'])

    # Calculate the contribution value for IV
    crosstab['contribution'] = (crosstab['p_good']-crosstab['p_bad'])*crosstab['WOE']

    # Calculate the IV
    IV = crosstab['contribution'][:-1].sum()

    add_IV = {'Characteristic': crosstab.index.name, 'Information Value': IV}

    WOE_list.append(crosstab)
    IV_list.append(add_IV)


# In[124]:


WOE_list


# In[125]:


# Create initial table to summarize the WOE values
WOE_table = pd.DataFrame({'Characteristic': [],
                          'Attribute': [],
                          'WOE': []})

for i in range(len(crosstab_list)):
    # Define crosstab and reset index
    crosstab = crosstab_list[i].reset_index()
    
    # Save the characteristic name
    char_name = crosstab.columns[0]
    
    # Only use two columns (Attribute name and its WOE value)
    # Drop the last row (average/total WOE)
    crosstab = crosstab.iloc[:-1, [0,-2]]
    crosstab.columns = ['Attribute', 'WOE']
    
    # Add the characteristic name in a column
    crosstab['Characteristic'] = char_name
    
    WOE_table = pd.concat((WOE_table, crosstab),
                        axis = 0)
    
    # Reorder the column
    WOE_table.columns = ['Characteristic',
                       'Attribute',
                       'WOE']
    
WOE_table


# In[126]:


WOE_table.info()


# In[127]:


# Put all IV in the table

IV_table = pd.DataFrame(IV_list)
IV_table


# ## Using WOE as Predictor Values

# In[128]:


# Function to generate the WOE mapping dictionary
def get_woe_map_dict(WOE_table):

    # Initialize the dictionary
    WOE_map_dict = {}
    WOE_map_dict['Missing'] = {}

    unique_char = set(WOE_table['Characteristic'])
    for char in unique_char:
        # Get the Attribute & WOE info for each characteristics
        current_data = (WOE_table
                            [WOE_table['Characteristic']==char]     # Filter based on characteristic
                            [['Attribute', 'WOE']])                 # Then select the attribute & WOE

        # Get the mapping
        WOE_map_dict[char] = {}
        for idx in current_data.index:
            attribute = current_data.loc[idx, 'Attribute']
            woe = current_data.loc[idx, 'WOE']

            if attribute == 'Missing':
                WOE_map_dict['Missing'][char] = woe
            else:
                WOE_map_dict[char][attribute] = woe
                WOE_map_dict['Missing'][char] = np.nan

    # Validate data
    print('Number of key : ', len(WOE_map_dict.keys()))

    return WOE_map_dict


# In[129]:


# Generate the WOE map dictionary
WOE_map_dict = get_woe_map_dict(WOE_table = WOE_table)
WOE_map_dict


# In[130]:


# Function to replace the raw data in the train set with WOE values
def transform_woe(raw_data, WOE_dict, num_cols):

    woe_data = raw_data.copy()

    # Map the raw data
    for col in woe_data.columns:
        if col in num_cols:
            map_col = col + '_bin'
        else:
            map_col = col

        woe_data[col] = woe_data[col].map(WOE_map_dict[map_col])

    # Map the raw data if there is a missing value or out of range value
    for col in woe_data.columns:
        if col in num_cols:
            map_col = col + '_bin'
        else:
            map_col = col

        woe_data[col] = woe_data[col].fillna(value=WOE_map_dict['Missing'][map_col])

    return woe_data


# In[131]:


data_train.info()


# In[132]:


# Transform the X_train
woe_train = transform_woe(raw_data = data_train.drop(['ID', 'status'], axis=1),
                          WOE_dict = WOE_map_dict,
                          num_cols = num_var)

woe_train = woe_train.drop(['CNT_CHILDREN_bin', 'AMT_INCOME_TOTAL_bin', 'AGE_bin', 'WORKING_YEARS_bin', 'CNT_FAM_MEMBERS_bin'], axis=1)
woe_train.head()


# In[133]:


woe_train.info()


# In[134]:


# Concatenate woe_train and y_train as data_train_use
data_train_use = pd.concat((woe_train, y_train),
                       axis = 1)


# In[135]:


data_train_use.info()


# # Test Data

# After doing EDA and cleansing on train data, we will do the same on test data but using train data characteristic to avoid any data leakage and creating condition as realistic as possibile to field condition.

# In[136]:


# Concatenate x_test and y_test as data_test
data_test = pd.concat((x_test, y_test), axis = 1)


# In[137]:


data_test.info()


# In[138]:


data_test.head()


# ## Own Car

# In[139]:


# Convert value into 1 and 0

data_test.loc[data_test['FLAG_OWN_CAR'] == "Y", 'FLAG_OWN_CAR'] = 1
data_test.loc[data_test['FLAG_OWN_CAR'] == "N", 'FLAG_OWN_CAR'] = 0


# ## Own Realty

# In[140]:


# Convert value into 1 and 0

data_test.loc[data_test['FLAG_OWN_REALTY'] == "Y", 'FLAG_OWN_REALTY'] = 1
data_test.loc[data_test['FLAG_OWN_REALTY'] == "N", 'FLAG_OWN_REALTY'] = 0


# ## Days Birth

# In[141]:


data_test['DAYS_BIRTH'] = data_test['DAYS_BIRTH']/365
data_test.rename(columns={'DAYS_BIRTH':'AGE'}, inplace=True)


# In[142]:


data_test.head()


# In[143]:


data_test.info()


# ## Days Employed

# In[144]:


data_test[data_test['DAYS_EMPLOYED']<1]


# In[145]:


data_test[data_test['DAYS_EMPLOYED']<1].info()


# In[146]:


data_test[data_test['DAYS_EMPLOYED']<1]['DAYS_EMPLOYED'].value_counts()


# In[147]:


len(data_test['DAYS_EMPLOYED'])


# In[148]:


data_test[data_test['DAYS_EMPLOYED']<1]['NAME_INCOME_TYPE'].value_counts()


# In[149]:


data_test['OCCUPATION_TYPE'].value_counts()


# In[150]:


# Handle errors

data_test.loc[data_test['DAYS_EMPLOYED'] < 1, 'OCCUPATION_TYPE'] = "Pensioner"
data_test.loc[data_test['DAYS_EMPLOYED'] < 1, 'DAYS_EMPLOYED'] = np.nan


# In[151]:


# Check missing values

data_test['DAYS_EMPLOYED'].isna().sum()


# In[152]:


# Convert into years

data_test['DAYS_EMPLOYED'] = data_test['DAYS_EMPLOYED']/365


# In[153]:


# Change column name

data_test.rename(columns={'DAYS_EMPLOYED':'WORKING_YEARS'}, inplace=True)


# In[154]:


data_test.info()


# ## Days Birth and Employed Analysis

# In[155]:


data_test[data_test['AGE']<data_test['WORKING_YEARS']]


# In[156]:


data_test[data_test['AGE']<data_test['WORKING_YEARS']]['WORKING_YEARS'].value_counts()


# In[157]:


data_test[data_test['AGE']<data_test['WORKING_YEARS']]['NAME_INCOME_TYPE'].value_counts()


# ## Occupation Type

# In[158]:


data_test.info()


# In[159]:


data_test['OCCUPATION_TYPE'].isna().sum()


# In[160]:


data_test[data_test['OCCUPATION_TYPE'].isna()]


# In[161]:


data_test['OCCUPATION_TYPE'] = data_test['OCCUPATION_TYPE'].fillna('Unknown')


# In[162]:


data_test['OCCUPATION_TYPE'].isna().sum()


# ## End of EDA

# In[163]:


data_test.info()


# In[164]:


# Change all flag-related columns into object type

data_test = data_test.astype({'FLAG_MOBIL':object, 'FLAG_WORK_PHONE':object, 'FLAG_PHONE':object, 'FLAG_EMAIL':object, 'FLAG_EMAIL':object})


# In[165]:


data_test.info()


# ## Binning

# In order to simplify our dataset and, more importantly, scaling all numerical variables into same scale with same weight, we do binning.

# In[166]:


for column in num_var:
    data_test_binned = create_binning(data = data_test,
                                    predictor_label = column,
                                    num_of_bins = 4)


# In[167]:


data_test_binned.T


# In[168]:


# Check for missing values

data_test_binned.isna().sum()


# In[169]:


# Turning empty values into a distinct category called 'Missing'

data_test_binned['WORKING_YEARS_bin'] = data_test_binned['WORKING_YEARS_bin'].cat.add_categories('Missing')
data_test_binned['WORKING_YEARS_bin'].fillna(value = 'Missing', inplace = True)


# In[170]:


# Check for missing values

data_test_binned.isna().sum()


# In[171]:


data_test.head()


# In[172]:


# Transform the X_train
woe_test = transform_woe(raw_data = data_test.drop(['ID', 'status'], axis=1),
                          WOE_dict = WOE_map_dict,
                          num_cols = num_var)
woe_test = woe_test.drop(['CNT_CHILDREN_bin', 'AMT_INCOME_TOTAL_bin', 'AGE_bin', 'WORKING_YEARS_bin', 'CNT_FAM_MEMBERS_bin'], axis=1)
woe_test.head()


# In[173]:


woe_test.info()


# In[174]:


# Concatenate woe_test and y_test as data_train_use
data_test_use = pd.concat((woe_test, y_test),
                       axis = 1)


# In[175]:


data_test_use.info()


# # Exporting Data

# In[176]:


# Export into csv file to use later

WOE_table.to_csv('../dataset/WOE_table_new.csv')
woe_train.to_csv('../dataset/woe_train.csv')
woe_test.to_csv('../dataset/woe_test.csv')
data_train_use.to_csv('../dataset/data_train_use_new.csv')
data_test_use.to_csv('../dataset/data_test_use_new.csv')


# # Scaling Data

# In[177]:


scaler_x_train = preprocessing.StandardScaler().fit(woe_train)


# In[178]:


scaler_x_train.mean_


# In[179]:


scaler_x_train.scale_


# In[180]:


x_train_scaled = scaler_x_train.transform(woe_train)
x_train_scaled


# In[181]:


x_train_scaled.mean(axis=0)


# In[182]:


x_train_scaled.std(axis=0)


# In[183]:


x_train_scaled = pd.DataFrame(x_train_scaled, columns=woe_train.columns)
x_train_scaled


# In[184]:


y_train = y_train.ravel()
y_train


# # Modelling

# ## Defining Params

# Params that will be used for Grid Search.

# In[185]:


params = {'C':[0.001, 0.1, 1, 10, 100, 1000]}


# ## Model 0 (Preliminary)

# This model uses no feature selection or, in another word, in its raw form.

# In[186]:


model_0 = LogisticRegression(random_state = 42, class_weight = 'balanced').fit(x_train_scaled, y_train)


# In[187]:


model_0.predict_proba(x_train_scaled)


# In[188]:


model_0.score(x_train_scaled, y_train)


# In[189]:


cv_result_0 = cross_validate(estimator = model_0,
                          X = x_train_scaled,
                          y = y_train,
                          scoring = 'roc_auc',
                          cv = 5)
cv_result_0['test_score']


# In[190]:


cv_mean_score_0 = np.mean(cv_result_0['test_score'])
cv_mean_score_0


# In[191]:


# Predict class labels for sample in x_train

y_train_pred_0 = model_0.predict(x_train_scaled)
y_train_pred_0


# In[192]:


# Calculate the recall score on the train set
recall_train_0 = recall_score(y_true = y_train,
                            y_pred = y_train_pred_0)

recall_train_0


# In[193]:


cm_0 = confusion_matrix(y_train, y_train_pred_0)
cm_0


# In[194]:


cmd_0 = ConfusionMatrixDisplay(cm_0)
cmd_0.plot()
plt.show()


# In[195]:


# Run GridSearch

model_0_gridsearch = GridSearchCV(estimator = LogisticRegression(random_state = 42, class_weight = 'balanced'),
                          param_grid = params,
                          cv = 5,
                          scoring = "roc_auc")


# In[196]:


model_0_gridsearch.fit(x_train_scaled, y_train)


# In[197]:


# Find the best params

model_0_gridsearch.best_params_


# In[198]:


# Find the best score

model_0_gridsearch.best_score_


# ## Model 1 (Forward Selection)

# In[199]:


# Define model
model_1 = LogisticRegression(random_state = 42, class_weight = 'balanced')


# In[200]:


model_1


# In[201]:


sfs1 = sfs(model_1, 
           k_features=(1,17), 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='roc_auc',
           cv=3)

sfs1 = sfs1.fit(x_train_scaled, y_train)


# In[202]:


model_1_features = sfs1.k_feature_idx_
model_1_features


# In[203]:


model_1_features = list(model_1_features)
model_1_features


# In[204]:


x_train_1 = x_train_scaled.iloc[:, model_1_features]
x_train_1


# In[205]:


model_1_a = LogisticRegression(random_state = 42, class_weight = 'balanced').fit(x_train_1, y_train)


# In[206]:


model_1_a.predict_proba(x_train_1)


# In[207]:


model_1_a.score(x_train_1, y_train)


# In[208]:


cv_result_1 = cross_validate(estimator = model_1_a,
                          X = x_train_1,
                          y = y_train,
                          scoring = 'roc_auc',
                          cv = 5)
cv_result_1['test_score']


# In[209]:


cv_mean_score_1 = np.mean(cv_result_1['test_score'])
cv_mean_score_1


# In[210]:


# Predict class labels for sample in x_train

y_train_pred_1 = model_1_a.predict(x_train_1)
y_train_pred_1


# In[211]:


# Calculate the recall score on the train set
recall_train_1 = recall_score(y_true = y_train,
                            y_pred = y_train_pred_1)

recall_train_1


# In[212]:


cm_1 = confusion_matrix(y_train, y_train_pred_1)
cm_1


# In[213]:


cmd_1 = ConfusionMatrixDisplay(cm_1)
cmd_1.plot()
plt.show()


# In[214]:


# Run GridSearch

model_1_gridsearch = GridSearchCV(estimator = LogisticRegression(random_state = 42, class_weight = 'balanced'),
                          param_grid = params,
                          cv = 5,
                          scoring = "roc_auc")


# In[215]:


model_1_gridsearch.fit(x_train_1, y_train)


# In[216]:


# Find the best params

best_params = model_1_gridsearch.best_params_
best_params


# In[217]:


# Find the best score
best_score = model_1_gridsearch.best_score_
best_score


# In[218]:


best_model = LogisticRegression(random_state = 42, class_weight = 'balanced', C=best_params['C']).fit(x_train_1, y_train)


# # Testing

# ## Test Data

# In[219]:


woe_test


# In[220]:


y_test


# ## Scaling Dataset

# In[221]:


scaler_x_test = preprocessing.StandardScaler().fit(woe_test)


# In[222]:


scaler_x_test.mean_


# In[223]:


scaler_x_test.scale_


# In[224]:


x_test_scaled = scaler_x_test.transform(woe_test)
x_test_scaled


# In[225]:


x_test_scaled.mean(axis=0)


# In[226]:


x_test_scaled.std(axis=0)


# In[227]:


x_test_scaled = pd.DataFrame(x_test_scaled, columns=woe_test.columns)
x_test_scaled


# In[228]:


x_test_scaled = x_test_scaled.iloc[:, model_1_features]
x_test_scaled


# In[229]:


y_test = y_test.ravel()
y_test


# ## Model

# In[230]:


model_test = LogisticRegression(random_state = 42, class_weight='balanced', C=best_params['C']).fit(x_test_scaled, y_test)


# In[231]:


model_test.predict_proba(x_test_scaled)


# In[232]:


model_test.score(x_test_scaled, y_test)


# In[233]:


cv_result_test = cross_validate(estimator = model_test,
                          X = x_test_scaled,
                          y = y_test,
                          scoring = 'roc_auc',
                          cv = 5)
cv_result_test['test_score']


# In[234]:


cv_mean_score_test = np.mean(cv_result_test['test_score'])
cv_mean_score_test


# In[235]:


# Predict class labels for sample in x_train

y_test_pred = model_test.predict(x_test_scaled)
y_test_pred


# In[236]:


# Calculate the recall score on the train set
recall_test = recall_score(y_true = y_test,
                            y_pred = y_test_pred)

recall_test


# In[237]:


cm_test = confusion_matrix(y_test, y_test_pred)
cm_test


# In[238]:


cmd_test = ConfusionMatrixDisplay(cm_test)
cmd_test.plot()
plt.show()


# # Scoring

# ## Model Summary

# In[239]:


intercept = best_model.intercept_
intercept


# In[240]:


best_model_intercept = pd.DataFrame({'Characteristic': 'Intercept',
                                     'Estimate': intercept})
best_model_intercept


# In[241]:


best_predictors = x_test_scaled.columns.tolist()
best_predictors


# In[242]:


coef = best_model.coef_
coef


# In[243]:


best_model_coefs = pd.DataFrame({'Characteristic':best_predictors,
                                 'Estimate':coef[0]})

best_model_summary = pd.concat((best_model_intercept, best_model_coefs),
                               axis = 0,
                               ignore_index = True)

best_model_summary


# ## Updating Model with Factor and Offset

# In[244]:


# Define Factor and Offset
factor = 20/np.log(2)
offset = 300-(factor*np.log(30))


# In[245]:


# Define n = number of characteristics
n = len(best_predictors)

# Define b0
b0 = intercept

n, b0


# In[246]:


# Define numerical predictors

num_var = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AGE', 'WORKING_YEARS', 'CNT_FAM_MEMBERS']


# In[247]:


# Load WOE_table

WOE_table


# ## Scorecards

# In[248]:


# Adjust characteristic name in best_model_summary_table
for col in best_predictors:
    if col in num_var:
        bin_col = col + '_bin'
    else:
        bin_col = col
    best_model_summary.replace(col, bin_col, inplace = True)


# Merge tables to get beta_i for each characteristic
scorecards = pd.merge(left = WOE_table,
                      right = best_model_summary,
                      how = 'left',
                      on = ['Characteristic'])

scorecards.head()


# In[249]:


scorecards.info()


# In[250]:


scorecards[scorecards['Estimate'].isna()]


# In[251]:


scorecards = scorecards.dropna()
scorecards.info()


# In[252]:


# Define beta and WOE

beta = scorecards['Estimate']
WOE = scorecards['WOE']

# Calculate the score point for each attribute
scorecards['Points'] = (offset/n) - factor*((b0/n) + (beta*WOE))

scorecards.head()


# In[253]:


# Calculate the min and max points for each characteristic
grouped_char = scorecards.groupby('Characteristic')
grouped_points = grouped_char['Points'].agg(['min', 'max'])
grouped_points


# In[254]:


# Calculate the min and max score from the scorecards
total_points = grouped_points.sum()
min_score = total_points['min']
max_score = total_points['max']

print(f"The lowest credit score = {min_score}")
print(f"The highest credit score = {max_score}")


# In[255]:


# Function to generate the points map dictionary
def get_points_map_dict(scorecards):

    # Initialize the dictionary
    points_map_dict = {}
    points_map_dict['Missing'] = {}
    unique_char = set(scorecards['Characteristic'])
    for char in unique_char:
        # Get the Attribute & WOE info for each characteristics
        current_data = (scorecards
                            [scorecards['Characteristic']==char]     # Filter based on characteristic
                            [['Attribute', 'Points']])               # Then select the attribute & WOE

        # Get the mapping
        points_map_dict[char] = {}
        for idx in current_data.index:
            attribute = current_data.loc[idx, 'Attribute']
            points = current_data.loc[idx, 'Points']

            if attribute == 'Missing':
                points_map_dict['Missing'][char] = points
            else:
                points_map_dict[char][attribute] = points
                points_map_dict['Missing'][char] = np.nan

    return points_map_dict


# In[256]:


# Generate the points map dict
points_map_dict = get_points_map_dict(scorecards = scorecards)
points_map_dict


# In[257]:


input = {'CODE_GENDER':'M',
 'FLAG_OWN_CAR':1,
 'FLAG_OWN_REALTY':0,
 'NAME_INCOME_TYPE':'Working',
 'NAME_EDUCATION_TYPE':'Higher education',
 'NAME_FAMILY_STATUS':'Married',
 'NAME_HOUSING_TYPE':'House / apartment',
 'FLAG_EMAIL':0,
 'OCCUPATION_TYPE':'Sales staff',
 'CNT_CHILDREN':2,
 'AMT_INCOME_TOTAL':100000,
 'AGE':30,
 'WORKING_YEARS':5,
 'CNT_FAM_MEMBERS':1}


# In[258]:


def transform_points(raw_data, points_map_dict, num_cols):

    points_data = raw_data.copy()

    # Map the data
    for col in points_data.columns:
        if col in num_cols:
            map_col = col + '_bin'
        else:
            map_col = col

        points_data[col] = points_data[col].map(points_map_dict[map_col])

    # Map the data if there is a missing value or out of range value
    for col in points_data.columns:
        if col in num_cols:
            map_col = col + '_bin'
        else:
            map_col = col

        points_data[col] = points_data[col].fillna(value=points_map_dict['Missing'][map_col])

    return points_data


# In[259]:


input_table = pd.DataFrame(input, index=[0])
input_points = transform_points(raw_data = input_table,
                                points_map_dict = points_map_dict,
                                num_cols = num_var)

input_points


# In[260]:


# Function to predict the credit score
def predict_score(raw_data, cutoff_score):

    # Transform raw input values into score points
    points = transform_points(raw_data = raw_data,
                              points_map_dict = points_map_dict,
                              num_cols = num_var)

    # Caculate the score as the total points
    score = int(points.sum(axis=1))

    print(f"Credit Score : ", score)

    if score > cutoff_score:
        print("Recommendation : APPROVE")
    else:
        print("Recommendation : REJECT")

    return score


# In[261]:


input_score = predict_score(raw_data = input_table,
                            cutoff_score = 205)


# In[270]:


# Transform the raw values in x_train into points
train_points = transform_points(raw_data = data_train.drop(['ID', 'status','FLAG_WORK_PHONE','FLAG_PHONE','FLAG_MOBIL', 'CNT_CHILDREN_bin', 'AMT_INCOME_TOTAL_bin', 'AGE_bin', 'WORKING_YEARS_bin', 'CNT_FAM_MEMBERS_bin'], axis=1),
                                points_map_dict = points_map_dict,
                                num_cols = num_var)

train_points.head()


# In[271]:


# Calculate the total score
train_points['SCORE'] = train_points.sum(axis=1).astype(int)
train_points.head()


# In[264]:


# Transform the raw values in x_test into points
# Excluding characteristics not used based on Model 1
test_points = transform_points(raw_data = data_test.drop(['ID', 'status', 'FLAG_WORK_PHONE','FLAG_PHONE','FLAG_MOBIL','CNT_CHILDREN_bin', 'AMT_INCOME_TOTAL_bin', 'AGE_bin', 'WORKING_YEARS_bin', 'CNT_FAM_MEMBERS_bin'], axis=1),
                                points_map_dict = points_map_dict,
                                num_cols = num_var)

test_points.head()


# In[273]:


# Calculate the total score
test_points['SCORE'] = test_points.sum(axis=1).astype(int)
test_points.head()


# In[274]:


# Distribution plot of predicted score
sns.histplot(x = test_points['SCORE'],
             kde = True)


# In[276]:


cutoff_list = []
approval_rate_list = []

for cutoff in range (int(min_score), int(max_score)):

  cutoff_list.append(cutoff)

  approve_counts = len(train_points[train_points['SCORE'] >= cutoff])
  n_sample = len(train_points)

  approval_rate = approve_counts/n_sample

  approval_rate_list.append(approval_rate)

approval_rate_table = pd.DataFrame({'Cutoff' : cutoff_list,
                                    'Expected Approval Rate' : approval_rate_list})

approval_rate_table


# In[277]:


# Plot the approval rate
plt.plot(approval_rate_table['Cutoff'],
         approval_rate_table['Expected Approval Rate'],
         label = "Expected Approval Rate")

plt.xlabel('Cutoff Score')
plt.ylabel('Approval Rate (Expected)')
plt.show()


# In[278]:


cutoff_list = []
bad_rate_list = []

for cutoff in range (int(min_score), int(max_score)):

  cutoff_list.append(cutoff)

  bad_counts = len(train_points[train_points['SCORE'] < cutoff])
  n_sample = len(train_points)

  bad_rate = bad_counts/n_sample

  bad_rate_list.append(bad_rate)

bad_rate_table = pd.DataFrame({'Cutoff' : cutoff_list,
                               'Expected Bad Rate' : bad_rate_list})

bad_rate_table


# In[279]:


# Plot the approval rate
plt.plot(approval_rate_table['Cutoff'],
         approval_rate_table['Expected Approval Rate'],
         label = "Expected Approval Rate")

# Plot the expected bad rate
plt.plot(bad_rate_table['Cutoff'],
         bad_rate_table['Expected Bad Rate'],
         label = "Expected Bad Rate",
         color = 'orange')

plt.ylabel("Expected Rate")
plt.xlabel("Cutoff Score")
plt.legend(loc = 1)
plt.show()


# In[ ]:




