# DM Project
Project implementation for the course of Data Mining at the University of Pisa 2020-2021. The project consists in the analysis of an unknown customer dataset.  
The project was done by [Simone Baccile](https://github.com/Simoniuss), [Lorenzo Simone](https://github.com/LorenzoSimone) and Marco Sorrenti.


 ##  Table of Contents
 * [Introduction](#introduction)
 * [Installation](#installation)


 ## Introduction
 The analysis of the dataset was divided in 4 steps:
 1. Data understanding
 2. Clustering analysis
 3. Predictive analysis
 4. Sequential pattern mining

 ### 1. Data understanding
 Analysis of the dataset, trying to understand what kind of dataset it is, what are the attributes, distributions of data. In this phase we've also done feature analysis, data cleaning and we've added new features useful to analyze the dataset in the following steps.

 ### 2. Clustering analysis
 In this step we've tested different clustering algorithm trying to understand better how data are distributed and how different customer can be classified. The algorithm tested are:
 * K-Means
 * Hierarchical clustering
 * DBSCAN
 * X-Means
 * G-Means

 ### 3. Predictive analysis
 During the predictive analysis we've used supervised learning algorithm to classify different kind of customers. Models used in this phase are:
 * Decision Tree
 * AdaBoost
 * Random Forest
 * K-NN
 * MLP

### 4. Sequential Pattern Mining
In this step we've tried to find statistically relevant patterns among data. The idea is to discover the hidden relationships between products and baskets in order to extract discriminatory behaviors. 
Pattern mining algorithm tested are:
* PrefixSpan
* GSP
* SPMF

More details about the project can be found in _DM_Report_ and in _Project Presentation_. 


 ## Installation
 Download the notebooks and create Python environment with Conda:
```
$ git clone https://github.com/Simoniuss/DM-Project
$ cd DM-Project
$ conda create -n DMProject python=3.8
$ conda activate DMProject
$ pip install -r requirements.txt
```