#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 10:30:40 2021

@author: amrithasubburayan case study 2
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv(r'//Users//amrithasubburayan//Downloads//casestudy.csv')

#•Total revenue for the current year
data['net_revenue'].sum(axis = 0)


#	New Customer Revenue 

#•	Existing Customer Revenue Current Year
d1 = data.loc[data['year'] == 2017, 'net_revenue'].sum()

#	Existing Customer Revenue Prior Year

d2 = data.loc[data['year'] == 2016, 'net_revenue'].sum()

d3 = data.loc[data['year'] == 2015, 'net_revenue'].sum()


#•	Existing Customer Growth
sum1 = d1 - d2
sum1 


#•	Total Customers Current Year 

data.loc[data['year'] == 2017, 'customer_email'].nunique()

#•	Total Customers Previous Year 
data.loc[data['year'] == 2016, 'customer_email'].nunique()