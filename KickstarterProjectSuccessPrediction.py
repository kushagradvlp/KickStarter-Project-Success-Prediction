import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Importing non graded and graded dataset
data_non_grade=pd.read_excel('Kickstarter.xlsx')

##Input the data here
data_grade=pd.read_excel('Kickstarter-Grading.xlsx')

#Removing project ids from non graded dataset that are already present in graded dataset
list_project_id=list(data_grade['project_id'])
data_non_grade=data_non_grade[~data_non_grade['project_id'].apply(lambda x: x in list_project_id)]
data_non_grade=data_non_grade[~data_non_grade['category'].isna()]



#First of all we filter the data for taking only the states that has successful and failed in them.
#Backers count should be removed as they will come in only after the launch of the project but we have to predict 
# the success when the project is launched
#On similar logic pledge, usd_pledged should also be removed
#Spotlight is highly correlated with state so we should remove this column
#disable_communication column has only one value so we should drop that column as it won't make any impact on the model
# As the timestamp columns are already segregated into day, date and hour, we can remove timestamp columns as well
#The categorical columns like country, currency, staff_pick, category needs to be dummified and the original 
#columns for those needs to be removed
#We can remove the project_id and name columns as they are unique for most of the rows and wouldn't make any effect
#while modeling

def data_preprocessing(df):
    df=df.drop(columns=['launch_to_state_change_days'])
    df=df.drop(columns=['spotlight'])
    df=df.drop(columns=['backers_count'])
    df=df[(df['state']=='successful') | (df['state']=='failed')]
    df=df.drop(columns=['name_len','blurb_len'])#,'name_len_clean','blurb_len_clean'])
    df=df.drop(columns=['state_changed_at_month', 'state_changed_at_yr'])
    labelencoder = LabelEncoder()
    df['state'] = labelencoder.fit_transform(df['state'])
    #df['country'] = labelencoder.fit_transform(df['country'])
    #df['currency'] = labelencoder.fit_transform(df['currency'])
    #df['staff_pick'] = labelencoder.fit_transform(df['staff_pick'])
    #df['category'] = labelencoder.fit_transform(df['category'])
    df=df.drop(columns=['disable_communication'])
    df=df.drop(columns=['pledged','usd_pledged','state_changed_at_day','state_changed_at_hr','state_changed_at_weekday'])
    df=df.drop(columns=['deadline','state_changed_at','created_at','launched_at']).dropna()
    project_id=list(df['project_id'])
    df=df.drop(columns=['name','project_id'])
    df=df.join(pd.get_dummies(df['country'],prefix='country'))
    df=df.join(pd.get_dummies(df['deadline_month'],prefix='deadline_month'))
    df=df.join(pd.get_dummies(df['created_at_month'],prefix='created_at_month'))
    df=df.join(pd.get_dummies(df['launched_at_month'],prefix='launched_at_month'))
    df=df.join(pd.get_dummies(df['deadline_weekday'],prefix='deadline_weekday'))
    df=df.join(pd.get_dummies(df['created_at_weekday'],prefix='created_at_weekday'))
    df=df.join(pd.get_dummies(df['launched_at_weekday'],prefix='launched_at_weekday'))
    df=df.join(pd.get_dummies(df['currency'],prefix='currency'))
    df=df.join(pd.get_dummies(df['staff_pick'],prefix='staff_pick'))
    df=df.join(pd.get_dummies(df['category'],prefix='category'))
    df=df.drop(columns=['country','currency','staff_pick','category','created_at_weekday','launched_at_weekday','deadline_weekday','deadline_month','created_at_month','launched_at_month'])#,'deadline_weekday','created_at_weekday','launched_at_weekday'])
    return df,project_id

#Dividing the non graded data and graded data in different dataframes and running the data preprocessing on
#both of the datasets
df_train,project_id=data_preprocessing(data_non_grade)
df_test,project_id=data_preprocessing(data_grade)


##To remove anomalies from the data, I used Isolation forest with contamination 0.04 to remove 4% of anomalies from
#the data
from sklearn.ensemble import IsolationForest
from numpy import where
import numpy as np
iforest=IsolationForest(contamination=.04)
pred=iforest.fit_predict(df_train)
score=iforest.decision_function(df_train)
anomaly_index=where(pred==-1)
list_remove=list(df_train.iloc[anomaly_index].index)
len(list_remove)
df_train.drop(list_remove, inplace=True)
df_train=df_train.reset_index().drop(columns=['index'])

# Divided the train and test set such that training has all the columns except for state and testing data just has 
#state
X_train=df_train[[cols for cols in df_train.columns if 'state' not in cols]]
X_test=df_test[[cols for cols in df_test.columns if 'state' not in cols]]
y_train=df_train['state']
y_test=df_test['state']


###Hyperparameter Tuning
'''
X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.3,random_state=42)
gbt=GradientBoostingClassifier(learning_rate= 0.1, min_samples_split= 2, n_estimators= 300, warm_start= False,max_depth=3,max_features='auto')
parameters = { "learning_rate": [0.3,0.2,0.1],
               "warm_start": [True,False],
               "min_samples_split": [1,2,3],
               "n_estimators": [100,150,200,250,300],
               "max_depth":[1,2,3,4,5],
               "max_features":['auto', 'sqrt', 'log2']
             }

grid = GridSearchCV(gbt, param_grid = parameters,verbose=1,n_jobs=2,cv = 10)
grid.fit(X_train,y_train)
print(grid.best_params_)
print(grid.best_score_)
'''

#the best params come out to be learning_rate= 0.1, min_samples_split= 2, n_estimators= 100, warm_start= False,
#max_depth=3,max_features='auto' 

## To select important features we did feature selection with random forest feature importance library.
## whichever feature has more importance than 1% i.e. 0.01 we took it as important
model=RandomForestClassifier()
model.fit(X_train, y_train)
df_importance=pd.DataFrame.from_dict(dict(zip(list(X_train.columns),model.feature_importances_)),orient='index').reset_index()
df_importance.columns=['features','importance']
df_importance.sort_values(by='importance',ascending=False)
imp_cols=list(df_importance[df_importance['importance']>=0.01].features)


# for mlp classifier we have standardized all the columns
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

import time

model_list=[]
acc_scr=[]
recall_scr=[]
prec_scr=[]
f1_scr=[]
specificity=[]
roc_auc=[]
time_taken=[]
i=0
dict_classifiers = {
    "Gradient Boost": GradientBoostingClassifier(learning_rate= 0.1, min_samples_split= 2, n_estimators= 300, warm_start= False)
}

for model, model_instantiation in dict_classifiers.items():
    print('\n'+model)
    temp=model
    model_list.append(model)
    start = time.process_time()
    model = model_instantiation
    if(temp=="MLPClassifier" or temp=="KNeighborsClassifier"):
        model=model.fit(X_train_scaled,y_train)
    else:
        model=model.fit(X_train,y_train)
    if(temp=="Random Forest" or temp=="Gradient Boost" or temp=="Extra Trees"):
        print(model)
        print(dict(zip(list(X_train.columns),model.feature_importances_))) 
    time_taken.append(time.process_time() - start)
    score=model.score(X_train,y_train)
    if(temp=="MLPClassifier" or temp=="KNeighborsClassifier"):
        predictions_model=model.predict(X_test_scaled)
    else:
        predictions_model=model.predict(X_test)
    predictions_model=model.predict(X_test)
    average_precision = average_precision_score(y_test, predictions_model)
    acc_scr.append(accuracy_score(y_test, predictions_model))
    recall_scr.append(recall_score(y_test, predictions_model))
    prec_scr.append(precision_score(y_test, predictions_model))
    f1_scr.append(f1_score(y_test, predictions_model))
    tn, fp, fn, tp =confusion_matrix(y_test, predictions_model).ravel()
    specificity.append(round(float(tn)/(tn+fp)*100,2))
    roc_auc.append(round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])*100,2))
    i=i+1
    plot_confusion_matrix(model,X_test, y_test)
    plt.show()

df_acc=pd.DataFrame(list(zip(model_list,time_taken,acc_scr,recall_scr,prec_scr,f1_scr,specificity,roc_auc)),columns=['Model','Time taken','Accuracy','Recall','Precision','F1 Score','Specificity','AUC'])

df_acc.to_excel('Model_Comparision.xlsx',index=False)



# Clustering
#Elbow method to show optimal value of k
df_train,project_id=data_preprocessing(data_non_grade)
df_test,project_id=data_preprocessing(data_grade)
df=df_train
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
labels =km.labels_
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

from sklearn.metrics import silhouette_samples
silhouette = silhouette_samples(df, labels)

from sklearn.metrics import silhouette_score
print("silhoutte score",silhouette_score(df, labels))

#cols=['goal', 'state', 'static_usd_rate',
#       'create_to_launch_days', 'launch_to_deadline_days']

#Taken only numerical columns
cols=['goal', 'static_usd_rate', 'name_len_clean', 'blurb_len_clean',
       'created_at_yr',
        'create_to_launch_days', 'launch_to_deadline_days','state']
df=df[cols]
scaler=StandardScaler()
standardized_x=scaler.fit_transform(df)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
df = pca.fit_transform(standardized_x)
reduced_df = pd.DataFrame(df, columns=['PC1','PC2'])
plt.scatter(reduced_df['PC1'], reduced_df['PC2'], alpha=.1, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

kmeans=KMeans(n_clusters=4)
model=kmeans.fit(reduced_df)
labels=model.labels_
from sklearn.metrics import silhouette_samples
silhouette = silhouette_samples(df, labels)

from sklearn.metrics import silhouette_score
print("silhoutte score",silhouette_score(df, labels))

labels=model.predict(reduced_df)
reduced_df['cluster'] = labels
list_labels=labels.tolist()
count1=count2=count3=count4=0
for i in list_labels:
    if i==0:
        count1+=1
    elif i==1:
        count2+=1
    elif i==2:
        count3+=1
    elif i==3:
        count4+=1
u_labels=np.unique(labels)
print("Number of datapoints in cluster 1 (K Means):", count1)
print("Number of datapoints in cluster 2 (K Means):", count2)
print("Number of datapoints in cluster 3 (K Means):", count3)
print("Number of datapoints in cluster 4 (K Means):", count4)
for i in u_labels:
    plt.scatter(df[labels == i , 0] , df[labels == i , 1] )
plt.legend(u_labels)
plt.show()


# Create a data frame containing our centroids
# Clustering


df_train,project_id=data_preprocessing(data_non_grade)
df_test,project_id=data_preprocessing(data_grade)
df=df_train

df=df[cols]

scaler=StandardScaler()
standardized_x=scaler.fit_transform(df)
standardized_x=pd.DataFrame(standardized_x,columns=df.columns)
df=standardized_x
kmeans=KMeans(n_clusters=4)
model=kmeans.fit(df)
labels=model.labels_
from sklearn.metrics import silhouette_samples
silhouette = silhouette_samples(df, labels)
df_train['label']=labels
print('Accuracy Score',accuracy_score(df_train['state'], df_train['label']))

from sklearn.metrics import silhouette_score
print("silhoutte score",silhouette_score(df, labels))

labels=model.predict(df)
df['cluster'] = labels
list_labels=labels.tolist()
count1=count2=count3=count4=0
for i in list_labels:
    if i==0:
        count1+=1
    elif i==1:
        count2+=1
    elif i==2:
        count3+=1
    elif i==3:
        count4+=1
u_labels=np.unique(labels)
print("Number of datapoints in cluster 1 (K Means):", count1)
print("Number of datapoints in cluster 2 (K Means):", count2)
print("Number of datapoints in cluster 3 (K Means):", count3)
print("Number of datapoints in cluster 4 (K Means):", count4)
df_train['label']=labels
print('Accuracy Score',accuracy_score(df_train['state'], df_train['label']))

import plotly.express as px
from pandas.plotting import *
centroids = pd.DataFrame(kmeans.cluster_centers_)
import plotly.io as pio
pio.renderers.default = 'browser'
fig=px.parallel_coordinates(centroids,labels=df.columns,color=u_labels)
fig.show()