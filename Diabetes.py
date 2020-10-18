#!/usr/bin/env python
# coding: utf-8

#   ## Importing necessary libraries

# In[1]:


from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Lets start by reading the Diabetes Data 

# In[2]:


diabetes = pd.read_csv("C:\\my-github-projects\\Diabetes-Disease-Prediction\\diabetes.csv")


# In[3]:


## Display all the columns of the dataframe

pd.pandas.set_option('display.max_columns',None)


# In[4]:


diabetes.head()


# In[5]:


diabetes.info()


# In[6]:


diabetes.shape


# In[7]:


diabetes.describe().T


# ## Check if our Dataset have any Zero Values

# In[8]:


diabetes.isnull().sum()


# In[9]:


check=diabetes[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
check.isin([0]).any().any()


# In[10]:


check=check.replace(0,np.nan)
check.head()


# In[11]:


check.isnull()


# In[12]:


## 1 -step make the list of features which has missing values
features_with_na=[features for features in check.columns if check[features].isnull().sum()>1]
## 2- step print the feature name and the percentage of missing values

for feature in features_with_na:
    print(feature, np.round(check[feature].isnull().mean(), 4),  ' % missing values')


# In[13]:


sns.heatmap(check.isnull() , yticklabels=False , cbar=False , cmap='viridis')


# In[14]:


# proportion of diabetes patients (about 35% having diabetes)
diabetes.Outcome.value_counts()[1] / diabetes.Outcome.count()


# In[15]:


# To analyse feature-outcome distribution in visualisation
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

ROWS, COLS = 2, 4
fig, ax = plt.subplots(ROWS, COLS, figsize=(18,8) )
row, col = 0, 0
for i, feature in enumerate(features):
    if col == COLS - 1:
        row += 1
    col = i % COLS
    
#     diabetes[feature].hist(bins=35, color='green', alpha=0.5, ax=ax[row, col]).set_title(feature)  #show all, comment off below 2 lines
    diabetes[diabetes.Outcome==0][feature].hist(bins=35, color='red', alpha=0.5, ax=ax[row, col]).set_title(feature)
    diabetes[diabetes.Outcome==1][feature].hist(bins=35, color='yellow', alpha=0.7, ax=ax[row, col])
    
plt.legend(['No Diabetes', 'Diabetes'])
fig.subplots_adjust(hspace=0.3)


# In[16]:


sns.set_style('whitegrid')
print(diabetes.Outcome.value_counts())
sns.countplot('Outcome',data=diabetes).set_title('Diabetes Outcome')


# In[17]:


list_diabetes=[268,500]
list_labels=['Diabetic','Healthy']


# In[18]:


plt.axis('equal')
plt.pie(list_diabetes,labels=list_labels,radius=2,autopct="%0.1f%%",shadow=True)


# In[19]:


sns.distplot(diabetes['Glucose'],kde=True,color='darkred',bins=40)
sns.set()


# In[20]:


def plot_prob_density(diabetes_Glucose,diabetes_BloodPressure):
    plt.figure(figsize = (10, 7))

    unit = 1.5
    x = np.linspace(Glucose.min() - unit, Glucose.max() + unit, 1000)[:, np.newaxis]
    

    # Plot the data using a normalized histogram
    plt.hist(df_lunch, bins=10, density=True, label='Glucose', color='orange', alpha=0.2)
    plt.hist(diabetes_BloodPressure, bins=10, density=True, label='BloodPressure', color='navy', alpha=0.2)
   
    # Do kernel density estimation
    kd_Glucose = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(df_lunch)
    kd_BloodPressure = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(df_lunch)
    

    
    # Plot the estimated densty
    kd_vals_Glucose = np.exp(kd_Glucose.score_samples(x))
    kd_vals_BloodPressure = np.exp(kd_BloodPressure.score_samples(x))
    

    plt.plot(x, kd_vals_Glucose, color='orange')
    plt.plot(x, kd_vals_Glucose, color='navy')
    
    
    plt.axvline(x=x_start,color='red',linestyle='dashed')
    plt.axvline(x=x_end,color='red',linestyle='dashed')
    

    # Show the plots
    plt.xlabel(field, fontsize=15)
    plt.ylabel('Probability Density', fontsize=15)
    plt.legend(fontsize=15)
    plt.show()
    gc.collect()
    return kd_Glucose


# In[21]:


def get_probability(start_value, end_value, eval_points, kd):
    
    # Number of evaluation points 
    N = eval_points                                      
    step = (end_value - start_value) / (N - 1)  # Step size

    x = np.linspace(start_value, end_value, N)[:, np.newaxis]  # Generate values in the range
    kd_vals = np.exp(kd.score_samples(x))  # Get PDF values for each x
    probability = np.sum(kd_vals * step)  # Approximate the integral of the PDF
    return probability.round(4)


# In[22]:


plt.figure(figsize = (10, 7))
sns.distplot(diabetes['Glucose'], label='Glucose')
sns.distplot(diabetes['BloodPressure'], label='Blood Pressure')
plt.xlabel('Glucose,BloodPressure', fontsize=15)
plt.ylabel('Probability Density', fontsize=15)
plt.legend(fontsize=15)
plt.show()


# In[23]:


sns.distplot(diabetes['SkinThickness'],kde=False,color='darkred',bins=40)


# In[24]:


sns.distplot(diabetes['Insulin'],kde=False,color='darkred',bins=40)


# In[25]:


sns.distplot(diabetes['BMI'],kde=False,color='darkred',bins=40)


# In[26]:


sns.countplot(x='Pregnancies', data=diabetes)


# In[27]:


diabetes['BloodPressure'].hist(color='green',bins=40,figsize=(8,4))


# In[28]:


sns.distplot(diabetes['BloodPressure'],kde=False,color='darkred',bins=40)


# ## Univariate Analysis
# 

# In[29]:


diabetic=diabetes.loc[diabetes['Outcome']==1]
non_diabetic=diabetes.loc[diabetes['Outcome']==0]


# In[30]:


plt.plot(diabetic['Insulin'],np.zeros_like(diabetic['Insulin']),'o')
plt.plot(non_diabetic['Insulin'],np.zeros_like(non_diabetic['Insulin']),'o')
plt.xlabel('Insulin')
plt.show()


# 
# ## Bivariate Analysis

# In[31]:


sns.FacetGrid(diabetes,hue='Outcome',height=6).map(plt.scatter,'Glucose','Insulin').add_legend()
plt.show()


# ### Multivariate Analysis

# In[32]:


# to visualise pair plot
sns.pairplot(diabetes, hue='Outcome', plot_kws=dict(alpha=.3, edgecolor='none'), height=2, aspect=1.1)
plt.show()


# 
# ## Correlation Matrix
# A correlation matrix is a table showing correlation coefficients between sets of variables. Each random variable (Xi) in the table is correlated with each of the other values in the table (Xj). This allows you to see which pairs have the highest correlation.
# 
# 

# In[33]:


#Pearson Correlation Cofficient
diabetes.corr()


# In[34]:


mask = np.zeros_like(diabetes.corr())
traingle_indices=np.triu_indices_from(mask)
mask[traingle_indices]=True
mask


# In[35]:


plt.figure(figsize=(16,10))
sns.heatmap(diabetes.corr(),mask=mask, annot=True, annot_kws={"size" : 14})
sns.set_style('white')
plt.xticks(fontsize=10)
plt.yticks(fontsize=14)
plt.show()


# ## Replacing Missing Values Inside Data

# In[36]:


diabetes['Glucose'].mean()


# In[37]:


diabetes['Glucose'].median()


# In[38]:


diabetes['Glucose'] = diabetes['Glucose'].replace(0, diabetes['Glucose'].median())
diabetes['BloodPressure'] = diabetes['BloodPressure'].replace(0, diabetes['BloodPressure'].median())
diabetes['SkinThickness'] = diabetes['SkinThickness'].replace(0, diabetes['SkinThickness'].median())
diabetes['Insulin'] = diabetes['Insulin'].replace(0, diabetes['Insulin'].median())   
diabetes['BMI'] = diabetes['BMI'].replace(0, diabetes['BMI'].median())
diabetes


# ## Visualization of Data After Removing Null Values

# In[39]:


sns.heatmap(diabetes.isnull() , yticklabels=False , cbar=False , cmap='viridis')


# ## Removing Outliers from the data

# In[40]:


diabetes.plot(kind='box',figsize=(20,10),color='Green',vert=False)
plt.show()


# In[41]:


SkinThickness_Outliers = diabetes['SkinThickness'].to_list()
Insulin_outliers = diabetes['Insulin'].to_list()


# In[42]:


outliers=[]
def detect_outliers(data):
    
    threshold=3
    mean = np.mean(data)
    std =np.std(data)
    
    
    for i in data:
        z_score= (i - mean)/std 
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return outliers


# In[43]:


outlier_pt=detect_outliers(SkinThickness_Outliers)
outlier_pt


# In[44]:


outlier_pt=detect_outliers(Insulin_outliers)
outlier_pt


# In[45]:


diabetes=diabetes[diabetes['SkinThickness']<80]
diabetes=diabetes[diabetes['Insulin']<=600]
print(diabetes.shape)


# ## Data Modelling
# We will work with six classification Algorithms.
# 1. KNN
# 2. Decision trees
# 3. Logistic Regression
# 4. SVM
# 5. Naive Bayes
# 6. Random Forest

# ## K-NEAREST NEIGHBOURS

# In[46]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(diabetes.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])


# In[47]:


X.head()


# In[48]:


#X = diabetes.drop("Outcome",axis = 1)
y = diabetes.Outcome


# ## Train Test Split Cross Validation methods

# In[61]:


#importing train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)


# In[62]:


from sklearn.neighbors import KNeighborsClassifier


test_scores = []
train_scores = []

for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))


# In[63]:


## score that comes from testing on the same datapoints that were used for training
max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))


# In[64]:


## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))


# # K Fold Cross Validation

# In[65]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(knn,X,y,cv=10)


# In[66]:


score


# In[67]:


score.mean()


# ## Stratified k fold cross validation

# In[68]:


X.shape,y.shape


# In[70]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42)


# In[93]:



len(y_test)


# In[94]:


len(y_train)


# In[90]:


np.count_nonzero(y_train)


# In[88]:


np.count_nonzero(y_test)


# In[72]:


X.iloc[600]


# In[75]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

accuracy=[]
skf=StratifiedKFold(n_splits=10,random_state=None)
skf.get_n_splits(X,y)
# X is the feature set and y is the target
for train_index,test_index in skf.split(X,y):
    print("Train:",train_index,"Validation:",test_index)
    X1_train, X1_test = X.iloc[train_index],X.iloc[test_index]
    y1_train, y1_test = y.iloc[train_index],y.iloc[test_index]
    
    knn.fit(X1_train,y1_train)
    prediction=knn.predict(X1_test)
    score=accuracy_score(prediction,y1_test)
    accuracy.append(score)
    
print(accuracy)# Here we see for 9th block the accuracy is highest


# In[76]:


np.array(accuracy).mean()#accuracy is list which doesnt have attribute mean so converted into array first


# ## Result Visualisation

# In[53]:


plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')
p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')


# The best result is captured at k = 11 hence 11 is used for the final model
# 

# In[54]:


#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(11)

knn.fit(X_train,y_train)
knn.score(X_test,y_test)


# ## Model Performance Analysis
# 1. Confusion Matrix
# The confusion matrix is a technique used for summarizing the performance of a classification algorithm i.e. it has binary outputs.

# In[55]:


#import confusion_matrix
from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above
y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[56]:


y_pred


# In[57]:


y_pred = knn.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[58]:


#import classification_report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# ## ROC - AUC
# ROC (Receiver Operating Characteristic) Curve tells us about how good the model can distinguish between two things (e.g If a patient has a disease or no). Better models can accurately distinguish between the two. Whereas, a poor model will have difficulties in distinguishing between the two

# In[59]:


from sklearn.metrics import roc_curve
y_pred_proba = knn.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


# In[60]:


plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=11) ROC curve')
plt.show()


# In[61]:


#Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_proba)


# ## Hyper Parameter optimization
# Grid search is an approach to hyperparameter tuning that will methodically build and evaluate a model for each combination of algorithm parameters specified in a grid.
# 
# Let’s consider the following example:
# 
# Suppose, a machine learning model X takes hyperparameters a1, a2 and a3. In grid searching, you first define the range of values for each of the hyperparameters a1, a2 and a3. You can think of this as an array of values for each of the hyperparameters. Now the grid search technique will construct many versions of X with all the possible combinations of hyperparameter (a1, a2 and a3) values that you defined in the first place. This range of hyperparameter values is referred to as the grid.
# 
# Suppose, you defined the grid as: a1 = [0,1,2,3,4,5] a2 = [10,20,30,40,5,60] a3 = [105,105,110,115,120,125]
# 
# Note that, the array of values of that you are defining for the hyperparameters has to be legitimate in a sense that you cannot supply Floating type values to the array if the hyperparameter only takes Integer values.
# 
# Now, grid search will begin its process of constructing several versions of X with the grid that you just defined.
# It will start with the combination of [0,10,105], and it will end with [5,60,125]. It will go through all the intermediate combinations between these two which makes grid search computationally very expensive.

# In[62]:


#import GridSearchCV
from sklearn.model_selection import GridSearchCV
#In case of classifier like knn the parameter to be tuned is n_neighbors
param_grid = {'n_neighbors':np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X,y)

print("Best Score:" + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_))


# ## Decision Trees.

# In[63]:


# feature selection
feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
x = diabetes[feature_cols]
y = diabetes.Outcome


# In[64]:


# split data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state=1)


# In[65]:


X_train.shape


# In[66]:


Y_train.shape


# In[67]:


X_test.shape


# In[68]:


Y_test.shape


# In[69]:


# build model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier = classifier.fit(X_train, Y_train)


# In[70]:


y_pred = classifier.predict(X_test)
print(y_pred)


# In[71]:


y_pred = classifier.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(Y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[72]:


# accuracy
print("Accuracy:", metrics.accuracy_score(Y_test,y_pred))


# In[73]:


#import classification_report
from sklearn.metrics import classification_report
print(classification_report(Y_test,y_pred))


# In[74]:


from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')
dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())


# ## Logistic Regression

# In[75]:


from sklearn.linear_model import LogisticRegression
regressor=LogisticRegression()
regressor.fit(X_train,Y_train)


# In[76]:


y_pred=regressor.predict(X_test)
y_pred


# In[77]:


# accuracy
print("Accuracy:", metrics.accuracy_score(Y_test,y_pred))


# In[78]:


#import classification_report
from sklearn.metrics import classification_report
print(classification_report(Y_test,y_pred))


# In[79]:


y_pred = regressor.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(Y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[80]:


from sklearn.metrics import confusion_matrix,classification_report,roc_curve,accuracy_score,auc
fpr,tpr,_=roc_curve(Y_test,y_pred)
#calculate AUC
roc_auc=auc(fpr,tpr)
print('ROC AUC: %0.2f' % roc_auc)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(fpr,tpr,label='ROC curve(area= %2.f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# In[81]:


#Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(Y_test,y_pred)


# ## Support Vector Machine

# ## SVM with RBF kernal

# In[82]:


std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)


# In[83]:


from sklearn.svm import SVC
model=SVC(kernel='rbf')
model.fit(X_train,Y_train)


# In[84]:


y_pred=model.predict(X_test)


# In[85]:


accuracy_score(Y_test,y_pred)


# In[86]:


y_pred = model.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(Y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[87]:


print(classification_report(Y_test,y_pred))


# In[88]:


fpr,tpr,_=roc_curve(Y_test,y_pred)
#calculate AUC
roc_auc=auc(fpr,tpr)
print('AUC: %0.2f' % roc_auc)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(fpr,tpr,label='ROC curve(area= %2.f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# In[89]:


#Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(Y_test,y_pred)


# ## SVM with Linear Kernel

# In[90]:


model=SVC(kernel='linear')
model.fit(X_train,Y_train)


# In[91]:


y_pred=model.predict(X_test)


# In[92]:


accuracy_score(Y_test,y_pred)


# In[93]:


y_pred = model.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(Y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[94]:


print(classification_report(Y_test,y_pred))


# In[95]:


fpr,tpr,_=roc_curve(Y_test,y_pred)
#calculate AUC
roc_auc=auc(fpr,tpr)
print('ROC AUC: %0.2f' % roc_auc)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(fpr,tpr,label='ROC curve(area= %2.f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# ## Naive Bayes

# In[96]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)


# In[97]:


y_pred = classifier.predict(X_test)


# In[98]:


accuracy_score(Y_test,y_pred)


# In[99]:



from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(Y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[100]:


print(classification_report(Y_test,y_pred))


# In[101]:


fpr,tpr,_=roc_curve(Y_test,y_pred)
#calculate AUC
roc_auc=auc(fpr,tpr)
print('ROC AUC: %0.2f' % roc_auc)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(fpr,tpr,label='ROC curve(area= %2.f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# ## Random Forest

# In[102]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,Y_train)


# In[103]:


Y_pred=classifier.predict(X_test)
confusion_matrix(Y_test,Y_pred)


# In[104]:



from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[105]:


accuracy_score(Y_test,Y_pred)


# In[106]:


print(classification_report(Y_test,Y_pred))


# In[107]:


fpr,tpr,_=roc_curve(Y_test,Y_pred)
#calculate AUC
roc_auc=auc(fpr,tpr)
print('ROC AUC: %0.2f' % roc_auc)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(fpr,tpr,label='ROC curve(area= %2.f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# ## Plotting ROC Curve for SVM, LR and RF

# In[108]:


# split data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state=1)


# In[109]:


# Import the classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve, roc_auc_score

# Instantiate the classfiers and make a list
classifiers = [LogisticRegression(random_state=1),  
               SVC(kernel='rbf',probability=True),
               RandomForestClassifier(random_state=1)]

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

# Train the models and record the results
for cls in classifiers:
    model = cls.fit(X_train, Y_train)
    yproba = model.predict_proba(X_test)[::,1]
    
    fpr, tpr, _ = roc_curve(Y_test,  yproba)
    auc = roc_auc_score(Y_test, yproba)
    
    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)


# In[110]:


fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()


# In[111]:


fig.savefig('multiple_roc_curve.png')


# In[112]:


#Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(Y_test,yproba)


# In[113]:


print(classification_report(Y_test,Y_pred))


# In[ ]:





# In[ ]:




