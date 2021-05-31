import pandas as pd
import numpy as np

print('hello')

class impute:
    def __init__(self):
        self.imputer_dict_for_prod = {}
        
    def fit(self,loan_data):
        cat_df = loan_data.drop(['Loan_ID'],axis=1).iloc[:,0:-1].select_dtypes(exclude=['int64','float64'])
        num_df = loan_data.drop(['Loan_ID'],axis=1).iloc[:,0:-1].select_dtypes(include=['int64','float64'])
        mode_impute_dict = cat_df.mode().iloc[0]
        mean_impute_dict = dict(num_df.mean())
        
        self.imputer_dict_for_prod = {**mode_impute_dict, **mean_impute_dict}
        
        cat_df.fillna(cat_df.mode().iloc[0],inplace=True)
        num_df.fillna(num_df.mean(),inplace=True)
        
        loan_data = pd.concat([cat_df,num_df,loan_data[['Loan_ID','Loan_Status']]],axis=1)
        
        return loan_data
        
        
    def transform(self,df):
        for i,j in self.imputer_dict_for_prod.items():
            df[i].fillna(j, inplace=True) 

        return df
    
    