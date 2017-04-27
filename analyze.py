#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:41:27 2017

@author: arlittr
"""

import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

import hdbscan
import pandas as pd
import numpy as np
from datetime import datetime
import itertools
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def readCSV(filename):
	with open(filename,'rU') as f:
		reader = csv.reader(f)
		thisfile = []
		for r in reader:
			if r: thisfile.append(r[0])
	return thisfile

def getDistanceBetweenTeams(team1,team2):
    #team1 and team2 are numpy matrices where rows are members and cols are attributes
    rows1 = itertools.permutations(range(np.shape(team1)[0]))
    rows2 = itertools.permutations(range(np.shape(team2)[0]))
    dist = []
    for r1 in rows1:
        for r2 in rows2:
            dist.append(calculateTeamDistance(team1.iloc[r1,:],team2.iloc[r2,:]))
    return min(dist)

def calculateTeamDistance(scoresList1,scoresList2):
    distances = []
    norms = []
    for scores1,scores2 in zip(np.array(scoresList1),np.array(scoresList2)):
        #L2 Norm
#        distances.append(sum([(s1-s2)**2 for s1,s2 in zip(scores1,scores2)])**0.5)
        #L1 Norm
#        distances.append(sum([abs(s1-s2) for s1,s2 in zip(scores1,scores2)]))
        #cosine distance 
        distances.append(scipy.spatial.distance.cosine(scores1,scores2))

        norms.append(np.array(np.linalg.norm((scores1 - scores2), ord=np.inf)))
   
#    assert(sum(norms)-sum(comparisons)<1e-10)

    return sum(distances)

def createDistanceMatrix(data,key):
    # sort the dataframe
#    data.sort(columns=[key], inplace=True)
    # set the index to be this and don't drop
#    data.set_index(keys=[key], drop=False,inplace=True)
    # get a list of names
    teamIDs=data[key].unique().tolist()
    d = dict.fromkeys(teamIDs)
    for k in d.keys():
        d[k] = {}
#    arr = np.empty()
    for id1,id2 in itertools.product(teamIDs,teamIDs):
        team1data = data[data.TeamID.isin([id1])][['E','S','T','J']]
        team2data = data[data.TeamID.isin([id2])][['E','S','T','J']]
        d[id1][id2] = getDistanceBetweenTeams(team1data,team2data)
        #make the matrix symmetric (too lazy to do something more clever)
        d[id2][id1] = d[id1][id2]
    return d

def generate_rosters(data,performanceMeasure,team):
    #generate rosters based on who was present
    for pm in performanceMeasure:
        df = data.loc[data[performanceMeasure].isnull() == False]
    for t in team:
        df = df.loc[df[t]==True]
#    teamIDs=data['TeamID'].unique().tolist()
#    for t in teamIDs:
#        df.loc[df['TeamID']==t]
    return df

def combine_team_performance(data,performanceType,combinationFn,newCol='metric'):
    #aggregates each team's performance measure such that each team has the same performance
    #data = dataframe
    #performanceType = key for performance metric that we're combining
    #combinationFn = function handle for how we'll combine (e.g., numpy.mean)
    teamIDs=data['TeamID'].unique().tolist()
    data[newCol] = 0
    for t in teamIDs:
        data.loc[data['TeamID']==t,newCol] = combinationFn(data.loc[data['TeamID']==t][performanceType])
	
    return data

def generate_summary_stats(data):
    summary_stats = [('E',np.mean,'E_avg'),('E',np.std,'E_std'),
                     ('S',np.mean,'S_avg'),('S',np.std,'S_std'),
                     ('T',np.mean,'T_avg'),('T',np.std,'T_std'),
                     ('J',np.mean,'J_avg'),('J',np.std,'J_std')]
    for old_category,combinationFn,newCol in summary_stats:
        data = combine_team_performance(data,old_category,combinationFn,newCol)
    
    return data
    
if __name__ == "__main__":
    path = '/Volumes/SanDisk/Repos/personality_ideation/data/data.csv'
    data = pd.read_csv(path)
    
    #generate rosters based on who was present
    analysis_scenarios = [('Zoo QUALITY',['Team Zoo'],np.max),
                          ('Zoo QUANTITY',['Team Zoo'],np.sum),
                          ('Zoo VARIETY',['Team Zoo'],np.mean),
                          ('Farmer QUALITY',['Team Farmer'],np.max),
                          ('Farmer QUANTITY',['Team Farmer'],np.sum),
                          ('Farmer VARIETY',['Team Farmer'],np.mean),
                          ('Travel QUALITY',['Team Travel'],np.max),
                          ('Travel QUANTITY',['Team Travel'],np.sum),
                          ('Travel VARIETY',['Team Travel'],np.mean),
                          ('Supplies QUALITY',['Team Supplies'],np.max),
                          ('Supplies QUANTITY',['Team Supplies'],np.sum),
                          ('Supplies VARIETY',['Team Supplies'],np.mean),
                          ]
    performance_measures = ['Zoo QUALITY','Zoo QUANTITY','Zoo VARIETY',
                            'Farmer QUALITY','Farmer QUANTITY','Farmer VARIETY',
                            'Travel QUALITY','Travel QUANTITY','Travel VARIETY',
                            'Supplies QUALITY','Supplies QUANTITY','Supplies VARIETY']
    scenario_dict = {}
    for a in analysis_scenarios:
        scenario = generate_rosters(data,a[0],a[1])
        scenario = combine_team_performance(scenario,a[0],a[2])
        reduced_scenario = generate_rosters(scenario,a[0],a[1])
        reduced_scenario = generate_summary_stats(reduced_scenario)
        distances = createDistanceMatrix(reduced_scenario[['TeamID','E','S','T','J']],key='TeamID')
        full_distance_matrix = pd.DataFrame.from_dict(distances)
        scenario_dict[a[0]] = reduced_scenario
        
        #PCA Summary Stats
        pca = PCA(n_components=3,whiten=True)
        summary_stats = ['E_avg','E_std','S_avg','S_std','T_avg','T_std','J_avg','J_std']
        pca.fit(reduced_scenario[summary_stats])
        pca.components_
        print(pca.explained_variance_ratio_)
        k=0
        print(reduced_scenario.groupby('TeamID').first()['metric'])
        for c in pca.components_:
            x = pca.transform(reduced_scenario.groupby('TeamID').first()[summary_stats])[:,k].reshape(-1,1)
            y = reduced_scenario.groupby('TeamID').first()['metric']
            lr = linear_model.LinearRegression(fit_intercept=True,normalize=True)
#            lr.fit(c.reshape(-1,1),reduced_scenario.groupby('TeamID').first()['metric'])
            lr.fit(x,y)
            print('Regression Coefs',lr.coef_)
#            print('R^2',lr.score(c.reshape(-1,1),reduced_scenario.groupby('TeamID').first()['metric']))
            print('R^2',lr.score(x,y))
            print('Residuals',lr.residues_)
            

            plt.plot(pca.transform(reduced_scenario.groupby('TeamID').first()[summary_stats])[:,k],reduced_scenario.groupby('TeamID').first()['metric'],'*')

#            plt.plot(c.reshape(-1,1),lr.predict(c.reshape(-1,1)),color='red')
#            plt.plot(c.reshape(-1,1),lr.predict(np.transpose(pca.components_)),color='red')
            plt.xlabel('Component '+str(k))
            plt.ylabel('Performance')
            plt.title(a[0])
            plt.show()
            k+=1
        
        transformed_x = pca.transform(reduced_scenario.groupby('TeamID').first()[summary_stats])
        y = reduced_scenario.groupby('TeamID').first()['metric']
        

        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("PC0 vs PC1 vs Performance "+a[0],fontsize=14)
        ax.set_xlabel("PC0",fontsize=12)
        ax.set_ylabel("PC1",fontsize=12)
        ax.scatter(transformed_x[:,0],transformed_x[:,1],s=100,c=y,marker='*',cmap=cm.bwr)
        plt.show()
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(transformed_x[:,0],transformed_x[:,1],transformed_x[:,2],s=100,c=y,marker='*',cmap=cm.bwr)
        ax.set_title("PC0 vs PC1 vs PC2 vs Performance "+a[0],fontsize=14)
        ax.set_xlabel("PC0",fontsize=12)
        ax.set_ylabel("PC1",fontsize=12)
        ax.set_zlabel("PC2",fontsize=12)
        plt.show()
        
        #PLS Summary Stats
        pls = PLSRegression(n_components=3)
        summary_stats = ['E_avg','E_std','S_avg','S_std','T_avg','T_std','J_avg','J_std']
        full_team_data = reduced_scenario.groupby('TeamID').first()[summary_stats+performance_measures].dropna()
        full_team_stats = full_team_data[summary_stats]
        full_performance = full_team_data[performance_measures]
        pls.fit(full_team_stats,full_performance)
        print('x weights',pls.x_weights_)
        print('y weights',pls.y_weights_)
        print('x loadings',pls.x_loadings_)
        print('y loadings',pls.y_loadings_)
        print('x scores',pls.x_scores_)
        print('y scores',pls.y_scores_)
        
#        #PCA
#        pca = PCA(n_components=3)
#        pca.fit(full_distance_matrix)
#        pca.components_
#        print(pca.explained_variance_ratio_)
#        k=0
#        print(reduced_scenario.groupby('TeamID').first()['metric'])
#        for c in pca.components_:
#            lr = linear_model.LinearRegression(fit_intercept=True,normalize=True)
##            lr.fit(c.reshape(-1,1),reduced_scenario.groupby('TeamID').first()['metric'])
#            lr.fit(np.transpose(pca.components_),reduced_scenario.groupby('TeamID').first()['metric'])
#            print('Regression Coefs',lr.coef_)
##            print('R^2',lr.score(c.reshape(-1,1),reduced_scenario.groupby('TeamID').first()['metric']))
#            print('R^2',lr.score(np.transpose(pca.components_),reduced_scenario.groupby('TeamID').first()['metric']))
##            print('Residuals',lr.residues_)
#            plt.plot(c.reshape(-1,1),reduced_scenario.groupby('TeamID').first()['metric'],'*')
##            plt.plot(c.reshape(-1,1),lr.predict(c.reshape(-1,1)),color='red')
##            plt.plot(c.reshape(-1,1),lr.predict(np.transpose(pca.components_)),color='red')
#            plt.xlabel('Component '+str(k))
#            plt.ylabel('Performance')
#            plt.title(a[0])
#            plt.show()
#            k+=1
#            
#            plt.plot()
#
#        
#        #Linear regression on PCA
#        
#        
#        pls = PLSRegression(n_components=3)
#        pls.fit(full_distance_matrix,reduced_scenario.groupby('TeamID').first()['metric'])
#        print('x weights',pls.x_weights_)
#        print('y weights',pls.y_weights_)
#        print('x loadings',pls.x_loadings_)
#        print('y loadings',pls.y_loadings_)
#        print('x scores',pls.x_scores_)
#        print('y scores',pls.y_scores_)
    
    combined_scenarios = [['Zoo QUALITY','Farmer QUALITY','Travel QUALITY','Supplies QUALITY'],
                          ['Zoo QUANTITY','Farmer QUANTITY','Travel QUANTITY','Supplies QUANTITY'],
                           ['Zoo VARIETY','Farmer VARIETY','Travel VARIETY','Supplies VARIETY']]
    df = pd.DataFrame()
    for cs in combined_scenarios:
        this_combined_scenario = pd.DataFrame()
        for s in cs:
            this_scenario = scenario_dict[s].groupby('TeamID').first()
            this_combined_scenario = this_combined_scenario.append(this_scenario)
        
        #PCA Summary Stats
        pca = PCA(n_components=3,whiten=True)
        summary_stats = ['E_avg','E_std','S_avg','S_std','T_avg','T_std','J_avg','J_std']
        pca.fit(this_combined_scenario[summary_stats])
        pca.components_
        print(pca.explained_variance_ratio_)
        k=0
        print(reduced_scenario.groupby('TeamID').first()['metric'])
        for c in pca.components_:
            x = pca.transform(this_combined_scenario[summary_stats])[:,k].reshape(-1,1)
            y = this_combined_scenario['metric']
            lr = linear_model.LinearRegression(fit_intercept=True,normalize=True)
#            lr.fit(c.reshape(-1,1),reduced_scenario.groupby('TeamID').first()['metric'])
            lr.fit(x,y)
            print('Regression Coefs',lr.coef_)
#            print('R^2',lr.score(c.reshape(-1,1),reduced_scenario.groupby('TeamID').first()['metric']))
            print('R^2',lr.score(x,y))
            print('Residuals',lr.residues_)
            

            plt.plot(x,y,'*')

#            plt.plot(c.reshape(-1,1),lr.predict(c.reshape(-1,1)),color='red')
#            plt.plot(c.reshape(-1,1),lr.predict(np.transpose(pca.components_)),color='red')
            plt.xlabel('Component '+str(k))
            plt.ylabel('Performance')
            plt.title(cs)
            plt.show()
            k+=1
        
        transformed_x = pca.transform(this_combined_scenario[summary_stats])
        y = this_combined_scenario['metric']
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("PC0 vs PC1 vs Performance "+' '.join(cs),fontsize=14)
        ax.set_xlabel("PC0",fontsize=12)
        ax.set_ylabel("PC1",fontsize=12)
        ax.scatter(transformed_x[:,0],transformed_x[:,1],s=100,c=y,marker='*',cmap=cm.bwr)
        plt.show()
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(transformed_x[:,0],transformed_x[:,1],transformed_x[:,2],s=100,c=y,marker='*',cmap=cm.bwr)
        ax.set_title("PC0 vs PC1 vs PC2 vs Performance "+' '.join(cs),fontsize=14)
        ax.set_xlabel("PC0",fontsize=12)
        ax.set_ylabel("PC1",fontsize=12)
        ax.set_zlabel("PC2",fontsize=12)
        plt.show()
       
    
    reduced_data = generate_rosters(data,'Zoo QUALITY',['Team Zoo','Team Travel'])
    distances = createDistanceMatrix(data[['TeamID','E','S','T','J']],key='TeamID')
    full_distance_matrix = pd.DataFrame.from_dict(distances)
    full_distance_matrix.to_csv('/Volumes/SanDisk/Repos/personality_ideation/data/distance_matrix.csv',encoding='utf-8')
    
#    print('Clustering with HDBSCAN ' + str(datetime.now()))
#    clusterer = hdbscan.HDBSCAN(min_cluster_size=2,metric='precomputed')
##    cluster_labels = clusterer.fit_predict(np.array(full_distance_matrix))
#    clusterer.fit(np.array(full_distance_matrix))
#    cluster_labels = clusterer.labels_
#    
#    #visualization
##    clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis', 
##                                      edge_alpha=0.6, 
##                                      node_size=80, 
##                                      edge_linewidth=2)
##    clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
#    clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
##    cluster_labels = clusterer.fit_predict(data[['E','S','T','J']].dropna())
#    for c in cluster_labels:
#        print(c)
#        
#    
#    outpath = '/Volumes/SanDisk/Repos/personality_ideation/data/output.csv'
#    clustering_results_d = {'teamIDs':list(full_distance_matrix.columns.values),
#                            'clusters':cluster_labels}
#    clustering_results = pd.DataFrame(clustering_results_d)
#    clustering_results.to_csv(outpath,encoding='utf-8')

#    lr = linear_model.LinearRegression(fit_intercept=True,normalize=True)
#    pca = PCA(n_components=10)
#    pca.fit(full_distance_matrix)
#    pca.components_
#    pca.explained_variance_
	