#!/usr/bin/env python
# coding: utf-8

# # Bengaluru Apartments Analysis

# ### AIM
# ###### Understanding and developing a regression model for the house prices of various types of settlements in and around Bangaluru City based on the current real estate value (in rupees) and size (in sq. ft)

# ## Import relevant libraries

# In[2]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sb
import matplotlib.pyplot as plt
sb.set()


# ## Read the data

# In[3]:


raw_data=pd.read_csv('blore_apartment_data.csv')


# In[4]:


raw_data


# ## Check for NULL values

# In[5]:


raw_data.isnull().sum()


# In[6]:


data=raw_data.copy()


# ## Drop all null value rows

# In[7]:


data=data.dropna(axis=0)


# In[8]:


data.isnull().sum()


# ## Check Price data to get an average estimate for each entry

# In[9]:


PriceList=list(data['Price'])
PriceList


# ##### Separate each price entry into it's minimum and maximum value into two different lists

# In[10]:


def sep(li):
    minli=[]
    maxli=[]
    for i in range(len(li)):
        text=li[i]
        heads,sep,tails=text.partition('-')
        minli.append(heads)
        maxli.append(tails)
    return minli,maxli


# In[11]:


minli,maxli=sep(PriceList)
minli


# In[12]:


maxli


# ## Replacing 'L','K','Cr' into their respective numeric conversion

# In[13]:


def Converter(li):
    newli=[]
    for i in range(len(li)):
        if 'L' in li[i]:
            text = li[i]
            li[i] = li[i].replace('L',' ')
            li[i] = float(li[i])
            li[i] = li[i]*100000
            li[i] = int(li[i])
            newli.append(li[i])
            li[i] = str(li[i])
        elif 'K' in li[i]:
            text = li[i]
            li[i] = li[i].replace('K',' ')
            li[i] = float(li[i])
            li[i] = li[i]*1000
            li[i] = int(li[i])
            newli.append(li[i])
            li[i] = str(li[i])
        elif 'Cr' in li[i]:
            text = li[i]
            li[i] = li[i].replace('Cr',' ')
            li[i] = float(li[i])
            li[i] = li[i]*10000000
            li[i] = int(li[i])
            newli.append(li[i])
            li[i] = str(li[i])
        else:
            newli.append(li[i])
    return newli


# In[14]:


min_range=Converter(minli)
max_range=Converter(maxli)


# # Check the entries of both arrays

# In[15]:


min_range


# In[16]:


max_range


# In[17]:


data['Min_range']=min_range


# In[18]:


data['Max_range']=max_range


# In[19]:


data.drop('Price',axis=1)


# In[20]:


data[["Min_range", "Max_range"]] = data[["Min_range", "Max_range"]].apply(pd.to_numeric)


# In[21]:


data


# In[22]:


col=data.loc[:,'Min_range':'Max_range']


# In[23]:


data['Average Price']=col.mean(axis=1)


# In[24]:


data.dropna(axis=0)


# In[25]:


data.describe().round(1)


# ## Drop the unwanted columns now

# In[26]:


data.drop(['Price','Min_range','Max_range'],axis=1)


# # Same procedure to refine the data for 'Area'

# In[27]:


AreaList=list(data['Area'])
def rem(li):
    area_li=[]
    for i in range(len(li)):
        text=li[i]
        if 'sq.ft' in text:
            li[i]=text.replace('sq.ft','')
            area_li.append(li[i])
        else:
            area_li.append(li[i])
    return area_li


# In[28]:


new_area_list=rem(AreaList)


# In[29]:


new_area_list


# In[30]:


Min,Max=sep(new_area_list)
Max


# In[31]:


data['MinArea']=Min
data['MaxArea']=Max


# ## Convert the given values in the column 'MinArea' and 'MaxArea' to numeric data

# In[32]:


data[['MinArea','MaxArea']]=data[['MinArea','MaxArea']].apply(pd.to_numeric)


# In[33]:


data['Average Area']=data[['MinArea','MaxArea']].mean(axis=1)


# In[34]:


data=data.drop(['Price','Area','Min_range','Max_range','MinArea','MaxArea'],axis=1)


# In[35]:


data.head()


# # UNIT TYPE (To separately understand the unit type)

# In[36]:


UnitTypeList=list(data['Unit Type'])


# In[37]:


def BHK(li):
    newli = []
    for i in range(len(li)):
        if 'Plot' in li[i]:
            li[i] = str("0 Not:BHK/RK ") + li[i]
            newli.append(li[i])
        else:
            newli.append(li[i])
    return newli


# In[38]:


BHK1=BHK(UnitTypeList)


# In[39]:


def sep(li):
    list1=[]
    list2=[]
    for i in range(len(li)):
        text=li[i]
        heads,sep,tails=text.partition(' ')
        list1.append(heads)
        list2.append(tails)
    return list1,list2
    


# In[40]:


nounit,unit_type=sep(UnitTypeList)


# In[41]:


BHK_OR_NOT,model_type=sep(unit_type)


# In[42]:


data['No of rooms']=nounit
data['BHK OR NOT']=BHK_OR_NOT
data['Type of house']=model_type


# In[43]:


data.drop(['Unit Type'],axis=1)


# In[44]:


data['No of rooms']=data['No of rooms'].replace('4+','4')


# In[45]:


data['Type of house'].value_counts()


# In[46]:


data['No of rooms'].value_counts()


# In[47]:


df=data[['No of rooms','BHK OR NOT','Type of house','Average Price','Average Area']]


# # Linear Regression

# ##### Declare independent and dependent variables

# In[48]:


from sklearn.model_selection import train_test_split


# In[49]:


x1=df['Average Area']
y=df['Average Price']
x1_matrix=x1.values.reshape(-1,1)


# In[50]:


x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.2,random_state=0)
x_train=x_train.values.reshape(-1,1)
x_test=x_test.values.reshape(-1,1)


# #### Build the Linear Regression model

# In[51]:


reg=LinearRegression()
reg.fit(x_train,y_train)


# In[52]:


y_pred=reg.predict(x_test)


# In[53]:


plt.scatter(x_test,y_test)
plt.title('Bengaluru Apartment Analysis')
plt.xlabel('Area',size=20)
plt.ylabel('Price',size=20)
plt.plot(x_test,reg.predict(x_test),color='black')
plt.show()


# In[54]:


reg.score(x_train,y_train)


# In[ ]:





# In[ ]:




