from __future__ import division
import numpy as np
import math
import pandas as pd
from sklearn import linear_model

print("Suburb name: North Ryde") #North Ryde
#Sample house prices
#House prices data obtained at: http://www.yourinvestmentpropertymag.com.au/top-suburbs/nsw-2113-north-ryde.aspx
house=np.zeros(100)
y1=1430 #*1000 in 2015
#Assuming 2015's average annual growth as median year on year growth
#growth value samples for last 100 years are generated from a normal distribution with mean=Average annual growth in 2015
#and standard deviation=annual growth in 2015-average annual growth
mean_growthy1=0.0974 #Average annual growth
#sigma_growth=sqrt(annual growth in 2015-average annual growth)
sigma_growthy1=0.0776
growth_y1 = np.random.normal(mean_growthy1, sigma_growthy1, 100)
house[99]=y1
for i in range(1,99):
	house[99-i]=house[99-i+1]/(1+growth_y1[i]) #computing past years' (before 2015) house prices

#Sample unit prices over years
#Unit prices data obtained at: http://www.yourinvestmentpropertymag.com.au/top-suburbs/nsw-2113-north-ryde.aspx
unit=np.zeros(100)
y2=895 #*1000 in 2015
#Assuming 2015's average annual growth as median year on year growth
#Samples are generated from a normal distribution with mean=Average annual growth in 2015
#and standard deviation=annual growth in 2015-average annual growth
mean_growthy2=0.061 #Average annual growth
#sigma_growth=sqrt(annual growth in 2015-average annual growth)
sigma_growthy2=0.0068
growth_y2=np.random.normal(mean_growthy2, sigma_growthy2, 100)
unit[99]=y2
for i in range(1,99):
	unit[99-i]=unit[99-i+1]/(1+growth_y2[i]) #computing past years' (before 2015) unit prices

#Sample employment numbers over the years
#Employment numbers for north ryde from: http://www.planning.nsw.gov.au/~/media/Files/DPE/Reports/forecasting-the-distribution-of-stand-alone-office-employment-within-sydney-to-2035-2014-08.ashx
numemployed=np.zeros(100)
#2015's number of employed people is taken as median and assuming normally distributed data
#mean job growth and standard deviation of job growth is computed from the 5 data points 
#then samples of growth values for last 100 years are generated from the normal distribution
mean_growthx1=1.2 #*1000 annual growth
sigma_growthx1=1.18
growth_x1=np.random.normal(mean_growthx1, sigma_growthx1, 112)
numemployed[99]=38 #*1000 in 2015
for i in range(1,99):
	numemployed[99-i]=numemployed[99-i]/(1+growth_x1[i]) #computing past years' (before 2015) no. of jobs
#then extrapolate data to obtain no. of employed people for the next 12 years
test_x1=np.zeros(12)
test_x1[0]=numemployed[99]+growth_x1[100]*numemployed[99]
for i in range(1,11):
	test_x1[i]=test_x1[i]+growth_x1[100+i]*test_x1[i]

#historical walk score of north ryde
#obtained from https://www.walkscore.com/AU-NSW/Sydney/North_Ryde
walkscore=np.zeros(100)
growth_x2=np.zeros(112)
for i in (0,111): #due to absence of historical walk score data online
	growth_x2[i]=math.log(1/growth_x1[i]) #over years walk score increase rate=log(1/job growth rate)
walkscore[99]=62 #walkscore in 2015
for i in range(1,99):
	walkscore[99-i]=walkscore[99-i]/(1+growth_x2[i])

test_x2=np.zeros(12)
test_x2[0]=walkscore[99]+growth_x2[100]*walkscore[99]
for i in range(1,11):
	test_x2[i]=test_x2[i]+growth_x2[100+i]*test_x2[i]

#Linear regression for house prices prediction
d1={'numemployed':numemployed,'walkscore':walkscore}
d2={'houseprices':house}
X_train=pd.DataFrame(d1) #Input dataframe of linear regression
Y_train=pd.DataFrame(d2) #Output dataframe from linear regression
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y_train) #Fit data to model to compute linear regression coefficients

# The coefficients
print 'Coefficients: ', regr.coef_
d4={'numemployed':test_x1,'walkscore':test_x2}
X_test=pd.DataFrame(d4)
Y=regr.predict(X_test) #Compute predicted house prices for the next 12 years using inferred model
print "HOUSE PRICES-- 2017: ",Y[1]," 2018: ",Y[2]," 2022: ",Y[6]," 2027: ",Y[7]
#Next for the same inputs fit linear regression model for unit prices
d5={'unitprices':unit}
Y1_train=pd.DataFrame(d5)
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y1_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
Y1=regr.predict(X_test) #Compute predicted unit prices for the next 12 years for same test set
print "UNIT PRICES-- 2017: ",Y1[1]," 2018: ",Y1[2]," 2022: ",Y1[6]," 2027: ",Y1[7]