#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("train.csv")


# In[4]:


df.head(5)


# In[5]:


df.shape


# In[6]:


#listing down the columns

df.columns.values


# # Categorical Columns
# 
# * Survived
# * PClass
# * Sex
# * SibSp
# * Parch
# * Embarked
# 

# # Numerical columns
# 
# * Age 
# * Fare 
# * Passenger

# # Mixed Columns
# 
# * Name
# * Ticket
# * Cabin

# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# # Few Conclusions
# 
# 1. Missing value in Age,Cabin and Embarked columns.
# 2. More than 70 percent values are missing in cabin columns, will have to drop.
# 3. Few columns have inappropriate data types.

# In[9]:


# Dropping cabin column

df.drop(columns=['Cabin'],inplace=True)


# In[10]:


#Imputing missing values for age
# Strategy = mean

df['Age'].fillna(df['Age'].mean(), inplace = True)


# In[11]:


#Imputing missing values for embarked

#finding the most appeared value in embarked column

df['Embarked'].value_counts()

# S it is

df['Embarked'].fillna('S',inplace=True)


# In[12]:


#want to check one more thing

# should i change the SibSp and Parch to categories

df['SibSp'].value_counts()


# In[13]:


df['Parch'].value_counts()


# # Changing data type for the following cols
# 
#  * Survived(category)
#  * PClass(category)
#  * Sex(category)
#  * Age(int)
#  * Embarked(category)

# In[14]:


df['Survived']=df['Survived'].astype('category')
df['Pclass']=df['Pclass'].astype('category')
df['Sex']=df['Sex'].astype('category')
df['Age']=df['Age'].astype('int')
df['Embarked']=df['Embarked'].astype('category')


# In[15]:


df.info()


# In[16]:


df.describe()


# # Univariate & BivariateAnalysis

# In[25]:


# Countplot of the 'Survived' column
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df)
plt.title('Count of Survived')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

death_percent = round((df['Survived'].value_counts().values[0]/891)*100)

print("Out of 891 -- {}% people died in the accident".format(death_percent))


# In[28]:


# Countplot of the 'Pclass' column
plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', data=df)
plt.title('Count of Pclass')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.show()

# Print customized frequency counts and percentages for Pclass
pclass_counts = df['Pclass'].value_counts()
pclass_percentages = (pclass_counts / len(df)) * 100

print("\nFrequency Counts and Percentages for Pclass:")
display(pclass_counts)
display(pclass_percentages)


# In[29]:


# Countplot of the 'Sex' column
plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', data=df)
plt.title('Count of Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# Print customized frequency counts and percentages for Sex
sex_counts = df['Sex'].value_counts()
sex_percentages = (sex_counts / len(df)) * 100

print("\nFrequency Counts and Percentages for Sex:")
display(sex_counts)
display(sex_percentages)


# In[30]:


# Countplot of the 'SibSp' column
plt.figure(figsize=(8, 5))
sns.countplot(x='SibSp', data=df)
plt.title('Count of SibSp')
plt.xlabel('Number of Siblings/Spouses Aboard')
plt.ylabel('Count')
plt.show()

# Print customized frequency counts and percentages for SibSp
sibsp_counts = df['SibSp'].value_counts()
sibsp_percentages = (sibsp_counts / len(df)) * 100

print("\nFrequency Counts and Percentages for SibSp:")
display(sibsp_counts)
display(sibsp_percentages)


# In[31]:


#Countplot of the 'Parch' column
plt.figure(figsize=(8, 5))
sns.countplot(x='Parch', data=df)
plt.title('Count of Parch')
plt.xlabel('Number of Parents/Children Aboard')
plt.ylabel('Count')
plt.show()

# Print customized frequency counts and percentages for Parch
parch_counts = df['Parch'].value_counts()
parch_percentages = (parch_counts / len(df)) * 100

print("\nFrequency Counts and Percentages for Parch:")
display(parch_counts)
display(parch_percentages)


# In[32]:


# Countplot of the 'Embarked' column
plt.figure(figsize=(8, 5))
sns.countplot(x='Embarked', data=df)
plt.title('Count of Embarked')
plt.xlabel('Port of Embarkation')
plt.ylabel('Count')
plt.show()

# Print customized frequency counts and percentages for Embarked
embarked_counts = df['Embarked'].value_counts()
embarked_percentages = (embarked_counts / len(df)) * 100

print("\nFrequency Counts and Percentages for Embarked:")
display(embarked_counts)
display(embarked_percentages)


# In[33]:


# Distplot for 'Age'
sns.distplot(df['Age'])
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Density')
plt.show()

# Print skewness and kurtosis for 'Age'
print(f"Skewness of Age: {df['Age'].skew():.4f}")
print(f"Kurtosis of Age: {df['Age'].kurt():.4f}")


# In[37]:


# Boxplot for 'Age'
sns.boxplot(x=df['Age'])
plt.title('Box Plot of Age')
plt.xlabel('Age')
plt.show()


# In[39]:


# Just out of curiosity

print("People with age in Between 60 and 70 are",df[(df['Age']>60) & (df['Age']<70)].shape[0])
print("People with age greater than 70 and 75 are",df[(df['Age']>=70) & (df['Age']<=75)].shape[0])
print("People with age greater than 75 are",df[df['Age']>75].shape[0])

print('-'*50)

print("People with age between 0 and 1",df[df['Age']<1].shape[0])


# # Conclusion
# 
# * For all practical purposes age can be considered as normal distribution.
# * Deeper analysis is required for outlier detection.

# In[41]:


# Distplot for 'Fare'
sns.distplot(df['Fare'])
plt.title('Distribution of Fare')
plt.xlabel('Fare')
plt.ylabel('Density')
plt.show()

# Print skewness and kurtosis for 'Fare'
print(f"Skewness of Fare: {df['Fare'].skew():.4f}")
print(f"Kurtosis of Fare: {df['Fare'].kurt():.4f}")


# In[43]:


# Boxplot for 'Fare'
sns.boxplot(x=df['Fare'])
plt.title('Box Plot of Fare')
plt.xlabel('Fare')
plt.show()


# In[44]:


print("People with fare in between $200 and $300",df[(df['Fare']>200) & (df['Fare']<300)].shape[0])
print("People with fare greater than $300",df[df['Fare']>300].shape[0])


# # Conclusion
# 
# * Highly skewed data, a lot of people had cheaper tickets.
# * Outliers are there in the data.

# # Multivariate analysis

# In[45]:


# Relationship between Survived and Pclass (categorical features)
plt.figure(figsize=(8, 5))
sns.countplot(x=df['Pclass'], hue=df['Survived'], data=df)
plt.title('Survival Count by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# Crosstab of Survived and Pclass with percentages
print("\nSurvival Rate (%) by Pclass:")
survival_rate_pclass = pd.crosstab(df['Pclass'], df['Survived']).apply(lambda r: round((r/r.sum())*100,1),axis=1)
display(survival_rate_pclass)


# In[46]:


# Relationship between Survived and Sex (categorical features)
plt.figure(figsize=(8, 5))
sns.countplot(x=df['Sex'], hue=df['Survived'], data=df)
plt.title('Survival Count by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# Crosstab of Survived and Sex with percentages
print("\nSurvival Rate (%) by Sex:")
survival_rate_sex = pd.crosstab(df['Sex'], df['Survived']).apply(lambda r: round((r/r.sum())*100,1),axis=1)
display(survival_rate_sex)


# In[47]:


# Relationship between Survived and Embarked (categorical features)
plt.figure(figsize=(8, 5))
sns.countplot(x=df['Embarked'], hue=df['Survived'], data=df)
plt.title('Survival Count by Embarked')
plt.xlabel('Embarked')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# Crosstab of Survived and Embarked with percentages
print("\nSurvival Rate (%) by Embarked:")
survival_rate_embarked = pd.crosstab(df['Embarked'], df['Survived']).apply(lambda r: round((r/r.sum())*100,1),axis=1)
display(survival_rate_embarked)


# In[51]:


# Survival with Age
plt.figure(figsize=(10, 6))
sns.histplot(df[df['Survived']==0]['Age'], kde=True, color='skyblue', label='Did not Survive')
sns.histplot(df[df['Survived']==1]['Age'], kde=True, color='lightcoral', label='Survived')
plt.title('Distribution of Age by Survival')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[52]:


# Survival with Fare
plt.figure(figsize=(10, 6))
sns.histplot(df[df['Survived']==0]['Fare'], kde=True, color='skyblue', label='Did not Survive')
sns.histplot(df[df['Survived']==1]['Fare'], kde=True, color='lightcoral', label='Survived')
plt.title('Distribution of Fare by Survival')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[54]:


sns.pairplot(df)


# In[56]:


# Calculate the correlation matrix for numerical features
# Exclude non-numeric columns before calculating correlation
numerical_df = df.select_dtypes(include=['number'])
correlation_matrix = numerical_df.corr()

# Display the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()


# # Feature Engineering

# In[58]:


# We will create a new column name Family which will be the sum of SibSp and Parch cols

df['family_size'] = df['Parch'] + df['SibSp']


# In[59]:


df.sample(5)


# In[60]:


# Now we will engineer a new feature by the name of family type

def family_type(number):
    if number==0:
        return "Alone"
    elif number > 0 and number <=4:
        return "Medium"
    else:
        return "Large"


# In[61]:


df['family_type']= df['family_size'].apply(family_type)


# In[62]:


df.sample(5)


# In[63]:


# Dropping SibSp, Parch and family_size

df.drop(columns=['SibSp','Parch','family_size'],inplace=True)


# In[64]:


df.sample(5)


# In[65]:


pd.crosstab(df['family_type'],df['Survived']).apply(lambda r: round((r/r.sum())*100,1),axis=1)


# # Detecting outliers
# 
# ## Numerical Data
# * If the data is following normal distribution, anything beyond 3SD - mean + 3SD can be considered as an outlier
# * If the data does not follow normal distribution, using boxplot we can eliminate points beyond Q1 - 1.5 IQR and Q3 + 1.5 IQR
# 
# 
# ## Categorical data
# * If the col is highly imbalnced for eg male 10000 and female 2 then we can eliminate female

# In[66]:


# Handling outliers in age(Almost normal)
df=df[df['Age']<(df['Age'].mean() + 3 * df['Age'].std())]
df.shape


# In[68]:


# handling outliers from Fare column
# Finding quartiles
Q1= np.percentile(df['Fare'],25)
Q3= np.percentile(df['Fare'],75)

outlier_low=Q1 - 1.5 * (Q3 - Q1)
outlier_high=Q3 + 1.5 * (Q3 - Q1)

df=df[(df['Fare']>outlier_low) & (df['Fare']<outlier_high)]


# In[69]:


# One hot encoding
df.sample(4)
# Cols to be transformed are Pclass, Sex, Embarked, family_type
pd.get_dummies(data=df, columns=['Pclass','Sex','Embarked','family_type'], drop_first=True)


# In[70]:


df = pd.get_dummies(data=df, columns=['Pclass','Sex','Embarked','family_type'], drop_first=True)


# In[72]:


plt.figure(figsize=(15,6))
sns.heatmap(df.corr(numeric_only=True), cmap='summer', annot=True, fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()


# # Drawing Conclusions
# * Chance of female survival is higher than male survival
# * Travelling in Pclass 3 was deadliest
# * Somehow, people going to C survived more
# * People in the age range of 20 to 40 had a higher chance of not surviving
# * People travelling with smaller families had a higher chance of surviving the accident in comparison to people with large families and travelling alone

# In[73]:


# Save the final cleaned Titanic dataset
df.to_csv("titanic_cleaned.csv", index=False)

