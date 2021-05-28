
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

class Feature_engineering:
    def __init__(self):
        self.z=0
        
    def fit(self,loan_data):
        
        loan_data.drop(['Loan_ID'],axis=1,inplace=True)
        
        catgeorical_features = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
        for feature in catgeorical_features:
            loan_data[feature] = loan_data[feature].astype('category')
        
        X = loan_data.iloc[:,loan_data.columns!='Loan_Status']
        y = loan_data.iloc[:,loan_data.columns=='Loan_Status']
        
        self.ohe = OneHotEncoder().fit(X.select_dtypes('category'))
        catg_cols_transform = self.ohe.transform(X.select_dtypes('category')).toarray()
        self.catg_feat_names = X.select_dtypes('category').columns
        dfOneHot = pd.DataFrame(catg_cols_transform, columns = self.ohe.get_feature_names(self.catg_feat_names))
        loan_data_OHE = pd.concat([X, dfOneHot], axis=1).drop(self.catg_feat_names,axis=1)
        
        loan_data = pd.concat([loan_data_OHE,y],axis=1)
        
        return loan_data
        
        
    def transform(self,X):
        X.drop(['Loan_ID'],axis=1,inplace=True)
        catgeorical_features = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
        for feature in catgeorical_features:
            X[feature] = X[feature].astype('category')
            
        catg_cols_transform = self.ohe.transform(X.select_dtypes('category')).toarray()
        dfOneHot = pd.DataFrame(catg_cols_transform, columns = self.ohe.get_feature_names(self.catg_feat_names))
        loan_data_OHE = pd.concat([X, dfOneHot], axis=1).drop(self.catg_feat_names,axis=1)
        
        return loan_data_OHE
    