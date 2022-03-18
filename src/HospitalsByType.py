import pandas as pd 
import matplotlib.pyplot as plt
import requests
from scipy import stats
import statsmodels.api as sm
import numpy as np

class Hospitals:
    '''
    A set of methods to look at scaling laws of hospitals as afunction of populatio and population density.

    17/10 - This one class looks at fitting  a model to number of beds in any hospital VS population size.
    
    '''

    def __init__(self):
        '''
        read data and initialized
        '''
        
       
        self.data = pd.read_csv('../assets/Definitive_HealthCare_USA_Hospital_Beds.csv')
        # 
        url_county_population = 'https://api.census.gov/data/2019/pep/population?get=POP,DENSITY&for=county:*'
        self.county_pop = pd.DataFrame(requests.get(url_county_population).json()[1:],columns = ["POP","DENSITY","state","COUNTYFIPS"])
        self.county_pop['POP'] = self.county_pop['POP'].apply(lambda x: int(x))
        
        fips_index = self.county_pop.apply( lambda x : x.state + x.COUNTYFIPS ,axis=1) 
        self.county_pop.set_index(fips_index, inplace=True)
        self.set_type_id()
        # 
    def set_type_id(self):
        '''
        A method to set the type of hospital, and add a column in the main dataset.
        '''
        type = self.data.HOSPITAL_TYPE.unique()
        
        dict_type = {k:v for  k,v in zip(type,range(type.shape[0]))}
        self.data['type_id'] = self.data.HOSPITAL_TYPE.apply(lambda x: dict_type[x])
        return dict_type
        
    def get_type(self, type_):
        '''
        A method to return a subsset of the data set by type
        '''
        pass



        
    def set_owner_id(self):
        '''
        A method to set an owner id for the kind of 'owner' in the hospital location columns
        '''
        owner_kind = self.data.OWNER.unique()
        owner_code = range(owner_kind.shape[0])
        owner_code_dict = dict(zip(owner_kind,owner_code))
        self.data['Owner_id'] = self.data.OWNER.apply( lambda x : owner_code_dict.get(x))


    def fit_model(self, y, df):
        '''
        Method to fit OLS  linear model
        '''

        X = sm.add_constant(df)
        res = sm.OLS(y,X).fit()
        return res

    def lrtest( self, ll0, ll1,df= 1):
        '''
        Perform Log-Liklihood testing
        '''
        lr = -2*(ll0- ll1)
        pval = stats.chi2.sf(lr, df)
       
        print( 'LR test {:3f} and p value {:5f}'.format(lr, pval))
        return lr, pval

        
    




    def  pop2beds(self,state_code, all=0):
        '''
        return the population in the fips to the total number of beds and the state.
        '''
        # formatting
        state = 'All'
        data = self.data.query('BEDS !=-999').query('TYPE == "GENERAL ACUTE CARE"') # remove missingness and look at acute general care 
        if (all == 0):
            data = data[data.ST_FIPS == state_code]
            state = data.STATE.iloc[0]
        beds = data[['COUNTYFIPS','BEDS']].groupby('COUNTYFIPS').sum()
        return beds.merge(self.county_pop, how = 'inner',right_index=True,left_index=True)[['POP','BEDS']] #, state

    # def plot_pop2beds(self,df,state,f):
    #     '''
    #     plot a df on a log log scale and return parameters
    #     df -  data frame to fit model
    #     state -  the state we are looking at 
    #     f - model function (self.power_law, self.straight_line)
    #     '''
    #     params, cov = self.fit_model(df,f )
    #     fig, (ax0,ax1,ax2) = plt.subplots(3,1)
    #     df.plot(kind = 'scatter', x = 'POP',y = 'BEDS', loglog=True,title = state , ax = ax0) 
    #     ax0.plot(df.POP, f(df.POP, *params), linestyle = '--', color = 'red')
    #     ax1.plot(df.BEDS-self.power_law(df.POP, *params))
    #     ax2.hist(df.BEDS-self.power_law(df.POP, *params))
    #     fig.show()
    #     return params, np.sqrt(np.diag(cov)) , df.BEDS-self.power_law(df.POP, *params)

    

    def study_beds(self):
        '''
        A method to study the opendat dc dataset
        '''
        pass
       



## run 


h = Hospitals()
# df = h.pop2beds(19)
# res0 = h.fit_model(df.BEDS, df.POP) # fit linear model
# res1 = h.fit_model(np.log(df.BEDS), np.log(df.POP)) # fit power law model
# lr , p = h.lrtest(res0.llf,res1.llf,1) # LR testing 

  
