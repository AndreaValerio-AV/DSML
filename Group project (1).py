#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer


# In[79]:


sales = pd.read_excel('sales.xlsx')
sales


# In[80]:


sales.set_index ('CustomerID', inplace = True)


# In[81]:


#check for duplicates 
sales [sales.duplicated()]


# In[15]:


sales.drop_duplicates(inplace= True)


# In[16]:


#Basic exploration 


# In[17]:


sales.shape


# In[18]:


sales.columns


# In[19]:


sales.head(5)


# In[20]:


sales.tail(5)


# In[21]:


sales.info()


# we have:
#  - 6 float variables 
#  - 8 integer variables 
#  - 1 object variable 

# In[22]:


sales.describe()


# - `count` we can see that the variable 'Recency' and 'MntDrinks' don't have the 7008 rows filled, this means we have a problem with NA's.
# - relação estreita entre Drinks, desserts and entries 
# 

# In[82]:


sales.loc[sales[['Recency']].idxmax()] # inspect the max value 99, it could have been coded to not have NA 


# In[83]:


sales.loc[sales[['MntEntries']].idxmax()]


# In[25]:


sales.loc[sales[['MntDrinks']].idxmax()]


# In[26]:


sales.skew()


# - `Moderate skewness` : NumAppPurchases  and NumStorePurchases
# - `High skewness` : MntMeat&Fish, MntEntries, MntVegan&Vegetarian,MntDrinks,MntDesserts, MntAdditionalRequests, NumOfferPurchases, NumTakeAwayPurchases, NumAppVisitsMonth and   Complain 

# In[27]:


sales.kurt()


# - `high Kurtosis (>3): `
# MntEntries                
# MntVegan&Vegetarian      
# MntDrinks                 
# MntDesserts               
# MntAdditionalRequests    
# NumOfferPurchases 
# NumTakeAwayPurchases
# NumAppVisitsMonth         
# Complain                 

# VISUAL EXPLO

# In[86]:


import numpy as np
import matplotlib.pyplot as plt


# In[85]:


fig, [ax6,ax7] = plt.subplots(nrows=1, ncols = 2, figsize=(15,2)) 
ax6.hist(sales['Recency'])
ax7.hist(sales['Complain'])

fig, [[ax0,ax1],[ax2,ax3],[ax4,ax5]] = plt.subplots(nrows=3, ncols = 2, figsize=(15,8)) 
ax0.hist(sales['MntMeat&Fish'])
ax1.hist(sales['MntEntries'])
ax2.hist(sales['MntVegan&Vegetarian'])
ax3.hist(sales['MntDesserts'])
ax4.hist(sales['MntDrinks'])
ax5.hist(sales['MntAdditionalRequests'])

fig, [[ax8,ax9],[ax10,ax11],[ax12,ax13]] = plt.subplots(nrows=3, ncols = 2, figsize=(15,8))
ax8.hist(sales['NumOfferPurchases'])            
ax9.hist(sales['NumAppPurchases'])          
ax10.hist(sales['NumTakeAwayPurchases'])   
ax11.hist(sales['NumStorePurchases'])        
ax12.hist(sales['NumAppVisitsMonth'])   

#TITULOS E LABELS
ax6.set_title ('Recency')
ax7.set_title ('Complain')
ax0.set_title('MntMeat&Fish')
ax1.set_title('MntEntries')
ax2.set_title('MntVegan&Vegetarian')
ax3.set_title('MntDesserts')
ax4.set_title('MntDrinks')
ax5.set_title('MntAdditionalRequests')
ax8.set_title('NumOfferPurchases')            
ax9.set_title('NumAppPurchases')          
ax10.set_title('NumTakeAwayPurchases')   
ax11.set_title('NumStorePurchases')        
ax12.set_title('NumAppVisitsMonth')


#TITULOS E LABELS
#ax2.plot(covid_data['new_cases'])
#ax2.set_title('Number of Cases (Daily) )')
#ax2.set_xlabel('Date')
#ax2.set_ylabel('Number of Cases')


# In[30]:


######################################################
######Fazer análise pairing com "teoria" por trás ####

sales_subset = sales[['MntMeat&Fish','MntVegan&Vegetarian','MntEntries','MntDesserts','MntDrinks','MntAdditionalRequests']].copy()
sns.pairplot(sales_subset)


# In[88]:


sales_subset1 = sales[['NumOfferPurchases','NumAppPurchases','NumTakeAwayPurchases','NumStorePurchases','NumAppVisitsMonth']].copy()
sns.pairplot(sales_subset1)


# In[32]:


sales_corr = sales.corr(method = 'spearman')
figure = plt.figure(figsize=(16,10))


x_axis_labels = ['Recency ', 'Meat&Fish', 'Entries', 'Vegan&Vegetarian', 'Drinks', 'Desserts', 'Additional Req.',
                'Offer Purch','App Purch',  'TakeAway', 'Store Purch', 'App visits Month', 'Complain'] # labels for x-axis
y_axis_labels = ['Recency ', 'Meat&Fish', 'Entries', 'Vegan&Vegetarian', 'Drinks', 'Desserts', 'Additional Req.',
                'Offer Purch','App Purch',  'TakeAway', 'Store Purch', 'App visits Month', 'Complain'] # labels for y-axis

sns.heatmap(sales_corr, annot=True, fmt = '.1g', 
            xticklabels=x_axis_labels,yticklabels=y_axis_labels,
           linewidth=0.5)


# ##Correlation coefficients whose magnitude are between 0.9 and 1.0 indicate variables which 
# ##can be considered very highly correlated. Correlation coefficients whose magnitude are between 0.7 and 0.9 
# ##indicate variables which can be considered highly correlated. Correlation coefficients whose
# ##magnitude are between 0.5 and 0.7 indicate variables which can be considered moderately correlated.
# ##Correlation coefficients whose magnitude are between 0.3 and 0.5 indicate variables which have a low correlation.
# 
# -  `Recency:` has a low correlation with all of the variables 
# - `Offer Purchases:` has a low correlation with all of the variables 
# - `Complain:`has a low correlation with all of the variables 
# - the rest of the variables atre all moderately correlated between each other. 

# DATA Cleaning 
#   
#   -Outliers
# 

# In[89]:


fig, [ax6,ax7] = plt.subplots(nrows=1, ncols = 2, figsize=(15,3)) 
ax6.boxplot(sales['Recency'])
ax7.boxplot(sales['Complain'])


# In[34]:


fig, [[ax0,ax1],[ax2,ax3],[ax4,ax5]] = plt.subplots(nrows=3, ncols = 2, figsize=(15,8)) 
ax0.boxplot(sales['MntMeat&Fish'])
ax1.boxplot(sales['MntEntries'])
ax2.boxplot(sales['MntVegan&Vegetarian'])
ax3.boxplot(sales['MntDesserts'])
ax4.boxplot(sales['MntDrinks'])
ax5.boxplot(sales['MntAdditionalRequests'])

fig, [[ax8,ax9],[ax10,ax11],[ax12,ax13]] = plt.subplots(nrows=3, ncols = 2, figsize=(15,8))
ax8.boxplot(sales['NumOfferPurchases'])            
ax9.boxplot(sales['NumAppPurchases'])          
ax10.boxplot(sales['NumTakeAwayPurchases'])   
ax11.boxplot(sales['NumStorePurchases'])        
ax12.boxplot(sales['NumAppVisitsMonth']) 


#TITULOS E LABELS
#ax13.set_title ('Recency')
#ax7.set_title ('Complain')
ax0.set_title('MntMeat&Fish')
ax1.set_title('MntEntries')
ax2.set_title('MntVegan&Vegetarian')
ax3.set_title('MntDesserts')
ax4.set_title('MntDrinks')
ax5.set_title('MntAdditionalRequests')
ax8.set_title('NumOfferPurchases')            
ax9.set_title('NumAppPurchases')          
ax10.set_title('NumTakeAwayPurchases')   
ax11.set_title('NumStorePurchases')        
ax12.set_title('NumAppVisitsMonth')


# 

# In[90]:


sales_3std = 3 *(sales.std().round(2))
sales_3std


# In[91]:


#check if there is negative values in the variables 
(sales [['Recency','Complain','MntMeat&Fish','MntEntries','MntVegan&Vegetarian',
      'MntDesserts','MntDrinks', 'MntAdditionalRequests','NumOfferPurchases','NumAppPurchases','NumTakeAwayPurchases',
      'NumStorePurchases', 'NumAppVisitsMonth']] < 0).sum()


# In[92]:


subset_data=sales[sales["Recency"]>sales_3std['Recency']]
subset_data
#we have 915 outliers 


# In[94]:


subset_data=sales[sales["MntMeat&Fish"]>sales_3std['MntMeat&Fish']]
subset_data
# we have 338 outliers 


# In[97]:


subset_data1=sales[sales["MntVegan&Vegetarian"]>sales_3std['MntVegan&Vegetarian']]
subset_data1
#we have 254 outliers 


# In[98]:


subset_data1=sales[sales["MntDesserts"]>sales_3std['MntDesserts']]
subset_data1
#383 outliers


# In[99]:


subset_data1=sales[sales["MntDrinks"]>sales_3std['MntDrinks']]
subset_data1
#369 outliers


# In[101]:


subset_data1=sales[sales["MntAdditionalRequests"]>sales_3std['MntAdditionalRequests']]
subset_data1
#397 outliers


# In[102]:


subset_data1=sales[sales["NumOfferPurchases"]>sales_3std['NumOfferPurchases']]
subset_data1
#365 outliers 


# In[103]:


subset_data1=sales[sales["NumAppPurchases"]>sales_3std['NumAppPurchases']]
subset_data1
#1343 out


# In[104]:


subset_data1=sales[sales["NumTakeAwayPurchases"]>sales_3std['NumTakeAwayPurchases']]
subset_data1
#417 out


# In[106]:


subset_data1=sales[sales["NumStorePurchases"]>sales_3std['NumStorePurchases']]
subset_data1
#1243 out


# In[107]:


subset_data1=sales[sales["NumAppVisitsMonth"]>sales_3std['NumAppVisitsMonth']]
subset_data1
#417 out


# explore recency = 99 to see if it is a code 
# is Recency = 99 is coded ? max outliers matches with recency =99; 

# In[109]:


(sales [['Recency']] == 99).sum()


# In[110]:


sales_R99 =sales[sales["Recency"] == 99 ]
sales_R99
#recency=99 doesn't seem a code 


# In[112]:


amount = sales[['MntMeat&Fish','MntEntries','MntVegan&Vegetarian',
      'MntDesserts','MntDrinks', 'MntAdditionalRequests']]
number = sales[['NumOfferPurchases','NumAppPurchases','NumTakeAwayPurchases',
     'NumStorePurchases', 'NumAppVisitsMonth
                


# In[114]:


sales[(sales['MntMeat&Fish']==0)& (sales['MntEntries']==0)& (sales['MntVegan&Vegetarian']==0)& 
      (sales['MntDesserts']==0)& (sales['MntDrinks']==0)& (sales['MntAdditionalRequests']==0)&
     (sales['NumOfferPurchases']!=0)& (sales['NumAppPurchases']!=0)& (sales['NumTakeAwayPurchases']!=0)& 
      (sales['NumStorePurchases']!=0) & (sales['NumAppVisitsMonth']!=0)]


# # Missing data 

# In[116]:


sales.isna().sum()


# In[118]:


sales[sales['MntDrinks']==0]


# In[119]:


sales[pd.isna(sales['MntDrinks'])]


# In[122]:


sales


# In[123]:


sales.isna().sum()/len(sales)*100


# In[72]:


((23+28)/7000)*100


# In[127]:


sales_drinks = sales[['MntDrinks']]

imputer = KNNImputer(n_neighbors=1)
array_impute = imputer.fit_transform(sales_drinks) # this is an array
sales_drinks = pd.DataFrame(array_impute, columns = sales_drinks.columns)

print(sales_drinks)


# In[ ]:




