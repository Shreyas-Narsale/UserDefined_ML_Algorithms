
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def MarvellousHeadBrainPredicator():
    
    data=pd.read_csv("MarvellousHeadBrainDataset.csv")
    print("Size if data set:",data.shape)
    
    #Load Data
    X=data['Head Size(cm^3)'].values
    Y=data['Brain Weight(grams)'].values
    
    #Least Square Method
    mean_x=np.mean(X)
    mean_y=np.mean(Y)
    
    n=len(X)
    
    numerator=0
    denomentor=0
    
        #Equation of line is y=mx+c
    for i in range(n):
        numerator+=((X[i]-mean_x)*(Y[i]-mean_y))
        denomentor+=(X[i]-mean_x)**2
    
    m=numerator/denomentor
    
    c=mean_y-(m*mean_x)
    
    print("Slope of Regression Line:",m)
    print("Y Intercept of Regression line:",c)
    
    max_x=np.max(X)+100#Return the maximum along a given axis.
    min_x=np.min(X)-100#Return the min along a given axis.
    
    #Display plotting above points
    x=np.linspace(min_x,max_x,n)#Return evenly spaced numbers over a specified interval.
    
    y=c+m*x
    
    plt.plot(x,y,color='#58b970',label='Regression Line')
    
    plt.scatter(X,Y,color='#ef5423',label='scatter plot')
    
    plt.xlabel('Head Size(cm^3)')
    
    plt.ylabel('Brain Weight(grams)')
    
    plt.legend()
    plt.show()
    
    #Findout goodness of fit i.e. fit of Regression
    
    ss_t=0
    ss_r=0
    
    for i in range(n):
        y_pred=c+(m*X[i])
        ss_r+=(Y[i]-y_pred)**2
        ss_t+=(Y[i]-mean_y)**2
    
    r2=1-(ss_r/ss_t)
    
    print("r2 is:",r2)
    

def main():
    MarvellousHeadBrainPredicator()


if __name__=="__main__":
    main()