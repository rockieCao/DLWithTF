# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(456)
import matplotlib.pyplot as plt
import pandas as pd
from ggplot import *
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('../data/kaggle_ForestCoverTypePrediction/train.csv') #you may find the data on kaggle site
df = df.drop(['Id'], axis=1)
print(df.describe(include = 'all'))
feature_names = df.columns.values[0:-1]
print("feature_names:", feature_names)

train,test = train_test_split(df, test_size=0.2, random_state=456)
valid,test = train_test_split(test, test_size=0.5, random_state=999)

# EDA
needEDA = False
con = ['Elevation' , 'Aspect' , 'Slope', 'Horizontal_Distance_To_Hydrology' , 'Vertical_Distance_To_Hydrology' ,'Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points','Cover_Type']
con_variables = train[con]
if needEDA:
    for i in range(len(con)):
        g = ggplot(con_variables, aes(x="Cover_Type", y=con[i])) + geom_boxplot()+ggtitle("Box plot of Cover Type and "
                                                                                          + con[i])+theme_bw()
        print(g)

# Continuous Feature Correlation (Correlation Matrix)
needCorr = False
if needCorr:
    cor = con_variables.iloc[:, 0:10]
    cor_matrix = cor.corr(method='pearson', min_periods=1)
    #print(cor_matrix)
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(cor_matrix, cmap=plt.cm.Blues, alpha=0.8)
    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    ax.set_frame_on(False)
    ax.set_yticks(np.arange(10) + 0.5, minor=False)
    ax.set_xticks(np.arange(10) + 0.5, minor=False)
    ax.set_xticklabels(con[0:10], minor=False)
    ax.set_yticklabels(con[0:10], minor=False)
    plt.xticks(rotation=90)
    plt.show()

# more analysis
needMoreEDA = False
if needMoreEDA:
    ## the geographical factors, elevation and slope, are also correlated with each other, and can be viewed as having co-influence on cover types
    con_variables["Type"] = con_variables["Cover_Type"].apply(lambda x:str(x))
    g = ggplot(con_variables, aes(x='Elevation',y='Slope',color='Type'))+geom_point()+theme_bw()
    print(g)

    g = ggplot(con_variables, aes(x='Elevation',y='Slope',color='Type'))+geom_point()+theme_bw()+facet_wrap('Type')
    print(g)

    ## Since the afternoon Hill shades are always correlated with the variable of aspect, we plot the afternoon shade index with the aspect variable and see if there is certain patterns on it.
    g=ggplot(con_variables,aes(x='Aspect',y='Hillshade_Noon',color='Type')) +geom_point() +theme_bw()+facet_wrap('Type')
    print(g)

    g=ggplot(con_variables,aes(x='Aspect',y='Hillshade_3pm',color='Type')) +geom_point() +theme_bw()+facet_wrap('Type')
    print(g)

    g=ggplot(con_variables,aes(x='Horizontal_Distance_To_Hydrology',y='Vertical_Distance_To_Hydrology',color='Type')) +geom_point() +theme_bw()+facet_wrap('Type')
    print(g)

# feature selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
features = train.iloc[:,0:-1]
label = train.iloc[:,-1]
sklearn_model = ExtraTreesClassifier()
sklearn_model.fit(features, label)
model = SelectFromModel(sklearn_model, prefit=True)
new_features = model.transform(features)
print("new_features.shape:", new_features.shape)

valid_features = valid.iloc[:, 0:-1]
valid_label = valid.iloc[:, -1]
valid_data = model.transform(valid_features)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
Classifiers = [DecisionTreeClassifier(),LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),RandomForestClassifier(n_estimators=200)]

models = []
accuracies = []
for clf in Classifiers:
    fit = clf.fit(new_features, label)
    pred = fit.predict(valid_data)
    models.append(clf.__class__.__name__)
    accuracies.append(accuracy_score(valid_label, pred))
    print('Accuracy of '+models[-1]+' is '+str(accuracies[-1]))

# filter by feature std in trees
isFilterByStd = False

if isFilterByStd:
    from sklearn.metrics import confusion_matrix
    confusion_matrix(valid_label, pred)
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title("Feature importances by std")
    plt.bar(range(new_features.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
    plt.xticks(range(new_features.shape[1]), indices)
    plt.xlim([-1, new_features.shape[1]])
    plt.show()

isFilterByImportance = True

if isFilterByImportance:
    feature_name = np.array(["%d-%s"%(b,a) for a,b in zip(feature_names, range(len(feature_names)))])
    rf = RandomForestClassifier(n_estimators=100, random_state=101).fit(valid_features, valid_label)
    importances = np.mean([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    indices = np.argsort(importances)
    range_ = range(len(importances))

    plt.figure()
    plt.title("Feature Importance by Tree")
    plt.barh(range_, importances[indices], color='r', xerr=std[indices], alpha=0.4, align='center')
    plt.yticks(range(len(importances)), feature_name[indices])
    plt.ylim([-1, len(importances)])
    plt.xlim([0.0, 0.65])
    plt.show()
