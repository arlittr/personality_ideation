#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:41:27 2017

@author: arlittr
"""

import csv
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils import resample

import hdbscan
import pandas as pd
import numpy as np
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

def pca_bootstrapping(scenario,xcols,ycols,titlestr,this_pca,nsamples):
    
    this_pca.fit(scenario[xcols])
    full_data_loadings = this_pca.components_
    
    j=0
    d = dict.fromkeys(range(len(this_pca.components_)))
    for k in d.keys():
        d[k]={}
        d[k]['explainedvarratios'] = []
        d[k]['spearmancorrs'] = []
        d[k]['spearmanps'] = []
        d[k]['loadings'] = []
        
    for n in range(nsamples):
        sampled_scenario = this_scenario.sample(frac=1,replace=True)
        this_pca.fit(sampled_scenario[xcols])
        this_transformed_x_full = this_pca.transform(sampled_scenario[xcols])
        y = sampled_scenario[ycols]
        
        j=0
        for c in this_pca.components_:
            x_transformed_1pc = this_transformed_x_full[:,j].reshape(-1,1)
#            d[j]['explainedvarratios'].append(this_pca.explained_variance_ratio_[j])
#            d[j]['spearmancorrs'].append(scipy.stats.spearmanr(x_transformed_1pc,y)[0])
#            d[j]['spearmanps'].append(scipy.stats.spearmanr(x_transformed_1pc,y)[1])
            if sum((c-full_data_loadings[j])**2) > sum((c+full_data_loadings[j])**2):
                d[j]['loadings'].append(-c)
                d[j]['explainedvarratios'].append(-this_pca.explained_variance_ratio_[j])
                d[j]['spearmancorrs'].append(-scipy.stats.spearmanr(x_transformed_1pc,y)[0])
                d[j]['spearmanps'].append(-scipy.stats.spearmanr(x_transformed_1pc,y)[1])
            else:
                d[j]['loadings'].append(c)
                d[j]['explainedvarratios'].append(this_pca.explained_variance_ratio_[j])
                d[j]['spearmancorrs'].append(scipy.stats.spearmanr(x_transformed_1pc,y)[0])
                d[j]['spearmanps'].append(scipy.stats.spearmanr(x_transformed_1pc,y)[1])
            j+=1
    
    d2 = {}
    for k in d.keys():
        d2[k] = {}
        d2[k]['explainedvarratios_avg']=np.mean(np.array(d[k]['explainedvarratios']),axis=0)
        d2[k]['explainedvarratios_std']=np.std(np.array(d[k]['explainedvarratios']),axis=0)
        d2[k]['spearmancorrs_avg']=np.mean(np.array(d[k]['spearmancorrs']),axis=0)
        d2[k]['spearmancorrs_std']=np.std(np.array(d[k]['spearmancorrs']),axis=0)
        d2[k]['spearmanps_avg']=np.mean(np.array(d[k]['spearmanps']),axis=0)
        d2[k]['spearmanps_std']=np.std(np.array(d[k]['spearmanps']),axis=0)
        d2[k]['loadings_avg']=np.mean(np.array(d[k]['loadings']),axis=0)
        d2[k]['loadings_std']=np.std(np.array(d[k]['loadings']),axis=0)
    
    print(d[0]['spearmancorrs'])
    print(np.histogram(d[0]['spearmancorrs']))
    plt.hist(d[0]['spearmancorrs'])
    plt.show()
    print(d[0]['spearmancorrs'])
#    print(d[0]['loadings'])
    for k,v in d2.items():
        print(k,v)
    print(d2[0]['loadings_avg'])
    print(d2[0]['loadings_std'])
    return d2

def pca_thing(scenario_data,xcols,ycols,titlestr):
    #PCA Summary Stats
    pca = PCA(n_components=3,whiten=True)
    pca.fit(scenario_data[xcols])
    print('Explained Variance',pca.explained_variance_ratio_)
    k=0
    transformed_x_full = pca.transform(scenario_data[xcols])
    y = scenario_data[ycols]
    
    results = pd.DataFrame(columns=('Case Label',
                                    'Explained Variance Ratio',
                                    'RegressionCoefs',
                                    'Regression R^2',
                                    'SpearmanCorr',
                                    'SpearmanPvalue',
                                    'Loadings'))
    
    if type(titlestr) == type([]):
        titlestr = ' '.join(titlestr)
        
    #Linear fits for each individual component
    for c in pca.components_:
        x_transformed_1pc = transformed_x_full[:,k].reshape(-1,1)
        lr = linear_model.LinearRegression(fit_intercept=True,normalize=True)
        lr.fit(x_transformed_1pc,y)
        print('Regression Coefs',lr.coef_)
        print('R^2',lr.score(x_transformed_1pc,y))
        print('Spearman: ',scipy.stats.spearmanr(x_transformed_1pc,y))
        print('Component: ',c)
        results.loc[len(results)] = np.nan
        results.loc[len(results)-1,'Case Label'] = titlestr+' Component '+str(k)
        results.loc[len(results)-1,'Explained Variance Ratio'] = pca.explained_variance_ratio_[k]
        results.set_value(len(results)-1,'RegressionCoefs',lr.coef_)
        results.loc[len(results)-1,'Regression R^2'] = lr.score(x_transformed_1pc,y)    
        results.loc[len(results)-1,'SpearmanCorr'] = scipy.stats.spearmanr(x_transformed_1pc,y)[0]
        results.loc[len(results)-1,'SpearmanPvalue'] = scipy.stats.spearmanr(x_transformed_1pc,y)[1]
        results.set_value(len(results)-1,'Loadings',c) 
        
        if scipy.stats.spearmanr(x_transformed_1pc,y)[1] < 0.05:
            pca_bootstrapping(scenario_data,xcols,ycols,titlestr,pca,nsamples=10000)
        
        plt.plot(x_transformed_1pc,y,'*')
        plt.xlabel('Component '+str(k))
        plt.ylabel('Performance')
        plt.title(titlestr)
        plt.show()
        k+=1
    
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("PC0 vs PC1 vs Performance "+' '.join(cs),fontsize=14)
    ax.set_xlabel("PC0",fontsize=12)
    ax.set_ylabel("PC1",fontsize=12)
    ax.scatter(transformed_x_full[:,0],transformed_x_full[:,1],s=100,c=y,marker='*',cmap=cm.bwr)
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(transformed_x_full[:,0],transformed_x_full[:,1],transformed_x_full[:,2],s=100,c=y,marker='*',cmap=cm.bwr)
    ax.set_title("PC0 vs PC1 vs PC2 vs Performance "+' '.join(cs),fontsize=14)
    ax.set_xlabel("PC0",fontsize=12)
    ax.set_ylabel("PC1",fontsize=12)
    ax.set_zlabel("PC2",fontsize=12)
    plt.show()
    
    return results
 
def pls_thing(scenario_data,xcols,ycols,titlestr):
    #PLS Summary Stats
    pls = PLSRegression(n_components=3)
    pls.fit(scenario_data[xcols],scenario_data[ycols])
    k=0
    transformed_x_full = pls.transform(scenario_data[xcols])
    y = scenario_data[ycols]
    
    results = pd.DataFrame(columns=('Case Label',
                                    'Explained Variance Ratio',
                                    'RegressionCoefs',
                                    'Regression R^2',
                                    'SpearmanCorr',
                                    'SpearmanPvalue',
                                    'Loadings',
                                    'X Weights',
                                    'X Loadings',
                                    'X Scores'))
    
    if type(titlestr) == type([]):
        titlestr = ' '.join(titlestr)
        
    #Linear fits for each individual component
    for c in range(np.shape(pls.x_weights_)[1]):
        x_transformed_1pc = transformed_x_full[:,k].reshape(-1,1)
        lr = linear_model.LinearRegression(fit_intercept=True,normalize=True)
        lr.fit(x_transformed_1pc,y)
        print('Regression Coefs',lr.coef_)
        print('R^2',lr.score(x_transformed_1pc,y))
        print('Spearman: ',scipy.stats.spearmanr(x_transformed_1pc,y))
        print('Component: ',c)
        results.loc[len(results)] = np.nan
        results.loc[len(results)-1,'Case Label'] = titlestr+' Component '+str(k)
#        results.loc[len(results)-1,'Explained Variance Ratio'] = pls.explained_variance_ratio_[k]
        results.set_value(len(results)-1,'RegressionCoefs',lr.coef_)
        results.loc[len(results)-1,'Regression R^2'] = lr.score(x_transformed_1pc,y)    
        results.loc[len(results)-1,'SpearmanCorr'] = scipy.stats.spearmanr(x_transformed_1pc,y)[0]
        results.loc[len(results)-1,'SpearmanPvalue'] = scipy.stats.spearmanr(x_transformed_1pc,y)[1]
        results.set_value(len(results)-1,'X Weights',pls.x_weights_[:,k])
        results.set_value(len(results)-1,'X Loadings',pls.x_loadings_[:,k]) 
        results.set_value(len(results)-1,'X Scores',pls.x_scores_[:,k]) 
        
        plt.plot(x_transformed_1pc,y,'*')
        plt.xlabel('Component '+str(k))
        plt.ylabel('Performance')
        plt.title('PLS '+titlestr)
        plt.show()
        k+=1
        print(results)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("PLS PC0 vs PC1 vs Performance "+' '.join(cs),fontsize=14)
    ax.set_xlabel("PC0",fontsize=12)
    ax.set_ylabel("PC1",fontsize=12)
    ax.scatter(transformed_x_full[:,0],transformed_x_full[:,1],s=100,c=y,marker='*',cmap=cm.bwr)
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(transformed_x_full[:,0],transformed_x_full[:,1],transformed_x_full[:,2],s=100,c=y,marker='*',cmap=cm.bwr)
    ax.set_title("PLS PC0 vs PC1 vs PC2 vs Performance "+' '.join(cs),fontsize=14)
    ax.set_xlabel("PC0",fontsize=12)
    ax.set_ylabel("PC1",fontsize=12)
    ax.set_zlabel("PC2",fontsize=12)
    plt.show()
    
    print(results)
    return results    

def make_briggs_attitudes(data):
    #Briggs attitudes, normalized 0-1
    data['ES'] = (data.E + 1-data.J + 2*data.S) / 4
    data['EN'] = (data.E + 1-data.J + 2*(1-data.S)) / 4
    data['ET'] = (data.E + data.J + 2*data['T']) / 4
    data['EF'] = (data.E + data.J + 2*(1-data['T'])) / 4
    data['IS'] = ((1-data.E) + data.J + 2*data.S) / 4
    data['IN'] = ((1-data.E) + data.J + 2*(1-data.S)) / 4
    data['IT'] = ((1-data.E) + (1-data.J) + 2*data['T']) /4
    data['IF'] = ((1-data.E) + (1-data.J) + 2*(1-data['T'])) /4
    
    return data

def calculate_cognitive_mode_memberships(data):
    """
    #determine how well each team satisfies Wilde's affinity group criteria
    #score bounded 0-1
    #score of 0 indicates all team members have 0 strength of preference for all 8 modes
    #score of 1 indicates that for each of the 8 modes, at least one team member has 
    a maximum strength of preference for that mode
    """
    teamIDs=data['TeamID'].unique().tolist()
        
    for t in teamIDs:
        data.loc[data['TeamID']==t,'affinity_experiment'] = calculate_affinity_strength(data.loc[data['TeamID']==t]['ES'])
        data.loc[data['TeamID']==t,'affinity_ideation'] = calculate_affinity_strength(data.loc[data['TeamID']==t]['EN'])
        data.loc[data['TeamID']==t,'affinity_organization'] = calculate_affinity_strength(data.loc[data['TeamID']==t]['ET'])
        data.loc[data['TeamID']==t,'affinity_community'] = calculate_affinity_strength(data.loc[data['TeamID']==t]['EF'])
        data.loc[data['TeamID']==t,'affinity_knowledge'] = calculate_affinity_strength(data.loc[data['TeamID']==t]['IS'])
        data.loc[data['TeamID']==t,'affinity_imagination'] = calculate_affinity_strength(data.loc[data['TeamID']==t]['IN'])
        data.loc[data['TeamID']==t,'affinity_analysis'] = calculate_affinity_strength(data.loc[data['TeamID']==t]['IT'])
        data.loc[data['TeamID']==t,'affinity_evaluation'] = calculate_affinity_strength(data.loc[data['TeamID']==t]['IF'])
        
    return data

def calculate_affinity_strength(col):
    """
    #determine how well the given team satisfies Wilde's affinity group criteria
    #score bounded 0-1
    #score of 0 indicates all team members have 0 strength of preference for all 8 modes
    #score of 1 indicates that for each of the 8 modes, at least one team member has 
    a maximum strength of preference for that mode
    """ 
    
    return max(col)

def perform_dimensionality_reduction_analyses(data):
    results = pd.DataFrame(columns=('Case Label',
                                    'Explained Variance Ratio',
                                    'RegressionCoefs',
                                    'Regression R^2',
                                    'SpearmanCorr',
                                    'SpearmanPvalue',
                                    'Loadings'))
    
    pls_results = pd.DataFrame(columns=('Case Label',
                                    'Explained Variance Ratio',
                                    'RegressionCoefs',
                                    'Regression R^2',
                                    'SpearmanCorr',
                                    'SpearmanPvalue',
                                    'Loadings'))
    
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
        
        #PCA Summary Stats for indivisdual 
        summary_stats = ['E_avg','E_std','S_avg','S_std','T_avg','T_std','J_avg','J_std']
#        results = results.append(pca_thing(reduced_scenario.groupby('TeamID').first(),xcols=summary_stats,ycols=['metric'],titlestr=a[0]))
        
        #PLS
#        pls_results = pls_results.append(pls_thing(reduced_scenario.groupby('TeamID').first(),xcols=summary_stats,ycols=['metric'],titlestr=a[0]))

    #Combine data from all four design problems
    combined_scenarios = [['Zoo QUALITY','Farmer QUALITY','Travel QUALITY','Supplies QUALITY'],
                          ['Zoo QUANTITY','Farmer QUANTITY','Travel QUANTITY','Supplies QUANTITY'],
                           ['Zoo VARIETY','Farmer VARIETY','Travel VARIETY','Supplies VARIETY']]
    
    for cs in combined_scenarios:
        this_combined_scenario = pd.DataFrame()
        for s in cs:
            this_scenario = scenario_dict[s].groupby('TeamID').first()
            this_combined_scenario = this_combined_scenario.append(this_scenario)
        results = results.append(pca_thing(this_combined_scenario,xcols=summary_stats,ycols=['metric'],titlestr=cs))  
#        pls_results = pls_results.append(pls_thing(this_combined_scenario,xcols=summary_stats,ycols=['metric'],titlestr=cs))  
        
            
    results.to_csv('/Volumes/SanDisk/Repos/personality_ideation/data/pca_results.csv',encoding='utf-8')
    #TODO: Pull out the right loadings/weights/scores to interpret
    pls_results.to_csv('/Volumes/SanDisk/Repos/personality_ideation/data/pls_results.csv',encoding='utf-8')      
    
    #     #PLS Summary Stats
#        pls = PLSRegression(n_components=3)
#        summary_stats = ['E_avg','E_std','S_avg','S_std','T_avg','T_std','J_avg','J_std']
#        full_team_data = reduced_scenario.groupby('TeamID').first()[summary_stats+performance_measures].dropna()
#        full_team_stats = full_team_data[summary_stats]
#        full_performance = full_team_data[performance_measures]
#        pls.fit(full_team_stats,full_performance)
#        print('x weights',pls.x_weights_)
#        print('y weights',pls.y_weights_)
#        print('x loadings',pls.x_loadings_)
#        print('y loadings',pls.y_loadings_)
#        print('x scores',pls.x_scores_)
#        print('y scores',pls.y_scores_)
#        

##        
##        
##        pls = PLSRegression(n_components=3)
##        pls.fit(full_distance_matrix,reduced_scenario.groupby('TeamID').first()['metric'])
##        print('x weights',pls.x_weights_)
##        print('y weights',pls.y_weights_)
##        print('x loadings',pls.x_loadings_)
##        print('y loadings',pls.y_loadings_)
##        print('x scores',pls.x_scores_)
##        print('y scores',pls.y_scores_)
#    

       
    
#    reduced_data = generate_rosters(data,'Zoo QUALITY',['Team Zoo','Team Travel'])
#    distances = createDistanceMatrix(data[['TeamID','E','S','T','J']],key='TeamID')
#    full_distance_matrix = pd.DataFrame.from_dict(distances)
#    full_distance_matrix.to_csv('/Volumes/SanDisk/Repos/personality_ideation/data/distance_matrix.csv',encoding='utf-8')
    
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

def perform_cognitive_mode_membership_analyses(data):
    
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
        scenario_dict[a[0]] = reduced_scenario
        
    #Combine data from all four design problems
    combined_scenarios = [['Zoo QUALITY','Farmer QUALITY','Travel QUALITY','Supplies QUALITY'],
                          ['Zoo QUANTITY','Farmer QUANTITY','Travel QUANTITY','Supplies QUANTITY'],
                           ['Zoo VARIETY','Farmer VARIETY','Travel VARIETY','Supplies VARIETY']]
    
    for cs in combined_scenarios:
        this_combined_scenario = pd.DataFrame()
        for s in cs:
            this_scenario = scenario_dict[s]
            this_combined_scenario = this_combined_scenario.append(this_scenario)
        
        
        this_combined_scenario = make_briggs_attitudes(this_combined_scenario)
        this_combined_scenario = calculate_cognitive_mode_memberships(this_combined_scenario)
        this_combined_scenario = this_combined_scenario.groupby('TeamID').first()
        this_combined_scenario['total_affinity'] = this_combined_scenario[[d for d in this_combined_scenario.columns if d.startswith('affinity_')]].sum(axis=1)
        print('Total Affinity for ',cs)
        print(scipy.stats.spearmanr(this_combined_scenario['total_affinity'],this_combined_scenario['metric']))
        plt.scatter(this_combined_scenario['total_affinity'],this_combined_scenario['metric'])
        plt.show()
        
#        for mode_col in [d for d in this_combined_scenario.columns if d.startswith('affinity_')]:
#            
#            lr = linear_model.LinearRegression(fit_intercept=True,normalize=True)
#
#            lr.fit(this_combined_scenario[mode_col],this_combined_scenario['metric'])
#            scipy.stats.spearmanr(this_combined_scenario[mode_col],this_combined_scenario['metric'])

        
    
if __name__ == "__main__":
    path = '/Volumes/SanDisk/Repos/personality_ideation/data/data.csv'
    data = pd.read_csv(path)
    

#    perform_dimensionality_reduction_analyses(data)
    perform_cognitive_mode_membership_analyses(data)
	