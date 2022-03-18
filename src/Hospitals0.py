import pandas as pd 
import matplotlib.pyplot as plt
import requests
from scipy.optimize  import curve_fit
import numpy as np

class Hospitals:
    '''
    A set of methods to look at scaling laws of hospitals as afunction of populatio and population density.
    '''

    def __init__(self):
        '''
        read data and initialized
        '''
        self.data = pd.read_csv("../assets/us_hospital_locations.csv")
       
        self.data0 = pd.read_csv('../assets/Definitive_HealthCare_USA_Hospital_Beds.csv')
        # 
        url_county_population = 'https://api.census.gov/data/2019/pep/population?get=POP,DENSITY&for=county:*'
        self.county_pop = pd.DataFrame(requests.get(url_county_population).json()[1:],columns = ["POP","DENSITY","state","COUNTYFIPS"])
        self.county_pop['POP'] = self.county_pop['POP'].apply(lambda x: int(x))
        self.county_pop['DENSITY'] = self.county_pop['POP'].apply(lambda x: float(x))
        fips_index = self.county_pop.apply( lambda x : x.state + x.COUNTYFIPS ,axis=1) 
        self.county_pop.set_index(fips_index, inplace=True)
        # 

    def set_owner_id(self):
        '''
        A method to set an owner id for the kind of 'owner' in the hospital location columns
        '''
        owner_kind = self.data.OWNER.unique()
        owner_code = range(owner_kind.shape[0])
        owner_code_dict = dict(zip(owner_kind,owner_code))
        self.data['Owner_id'] = self.data.OWNER.apply( lambda x : owner_code_dict.get(x))

        
    
    def power_law(self,x,a,b):
        '''
        return  power law with constant a and b 
        '''
        return a*np.power(x,b)

    def straight_line(self,x ,a,b):
        '''
        return a straight line  for the kind ax +b 
        '''
        return a*x+b

    def fit_model(self,df,f ):
        '''
        A method to return params and cov for power law fitting for the data frame specificaly fiting bed numbers and population sizd
        '''
        params, cov = curve_fit(f = f, xdata = df.POP, ydata = df.BEDS, p0 = [0,0], bounds=[-np.inf, np.inf])

        return params, cov



    def get_gof(self,df,params,f):
        '''
        Perform a goodness of fit testing
        '''
        rss = (df.BEDS - f(df.POP, *params)).sum()
        tss = np.power(df.POP - np.ones(df.POP.shape[0])*df.POP.mean(),2).sum()
        ess = (np.power(f(df.POP, *params) - np.ones(df.POP.shape[0])*df.POP.mean() , 2)).sum()

        
        return {'rss':rss, 'tss':tss ,'ess':ess, 'r2':ess/float(tss)}






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
        return beds.merge(self.county_pop, how = 'inner',right_index=True,left_index=True)[['POP','BEDS']] , state

    def plot_pop2beds(self,df,state,f):
        '''
        plot a df on a log log scale and return parameters
        df -  data frame to fit model
        state -  the state we are looking at 
        f - model function (self.power_law, self.straight_line)
        '''
        params, cov = self.fit_model(df,f )
        fig, (ax0,ax1,ax2) = plt.subplots(3,1)
        df.plot(kind = 'scatter', x = 'POP',y = 'BEDS', loglog=True,title = state , ax = ax0) 
        ax0.plot(df.POP, f(df.POP, *params), linestyle = '--', color = 'red')
        ax1.plot(df.BEDS-self.power_law(df.POP, *params))
        ax2.hist(df.BEDS-self.power_law(df.POP, *params))
        fig.show()
        return params, np.sqrt(np.diag(cov)) , df.BEDS-self.power_law(df.POP, *params)

    

    def study_beds(self):
        '''
        A method to study the opendat dc dataset
        '''
        pass
       



## run 


h = Hospitals()
f = h.straight_line
df,state = h.pop2beds(21,all =0) 
params, cov = h.fit_model(df,f)
gof_dict = h.get_gof(df, params,f)
p,c,r = h.plot_pop2beds(df,state,f)  
