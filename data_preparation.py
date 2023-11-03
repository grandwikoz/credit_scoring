#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries

import pandas as pd
import numpy as np


# In[2]:


# Load dataset

application = pd.read_csv("../dataset/application_record.csv")
credit = pd.read_csv("../dataset/credit_record.csv")


# In[3]:


application.head()


# In[4]:


credit.head()


# In[5]:


credit_summary = credit.groupby(['ID', 'STATUS'], as_index=False).agg('count')
credit_summary = credit_summary.drop('MONTHS_BALANCE', axis=1)
credit_summary


# In[6]:


due_client = credit_summary[credit_summary['STATUS']=='1']
due_client = due_client['ID'].unique().tolist()
due_client


# In[7]:


client_status = []
for client in application['ID']:
    if client in due_client:
        value = 1
    else:
        value = 0
    client_status.append(value)


# In[8]:


client_status


# In[9]:


application['status'] = client_status


# In[10]:


application.head()


# In[11]:


application['status'].value_counts(normalize=True)

