
# coding: utf-8

# In[1]:


##############################
#Benjamin Stark 
#University of Montana
#Implementation of the E-M Algorithm for Univariate Gaussian Mixture Modeling. 
#This implementation is intended for UNIVARIATE DATA ONLY
##############################


# In[1]:


#import necessary libraries:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math


# In[12]:


##############################################
#Data Upload, this is a random data example.  
#upload other data in the block below
Normal_1=np.random.normal(loc=200.0, scale=15.0, size=20000)
Normal_2=np.random.normal(loc=115,scale=23,size=13000)
Data= np.concatenate((Normal_1,Normal_2),axis=0)
###############################################
#Upload data as one vector in the block below using your path: 
path='/Users/math/Desktop/M561/MSMS_Test1.txt'
Data = np.genfromtxt(path, delimiter='\n')
###############################################


# In[13]:


############################
#Visualize the Data. 
plt.hist(Data,bins='auto')
plt.show()


# In[14]:


################################
#Initialization step.
#Initialize the means, covariances, and mixing coefficients evaluate the intial likelihood function. 
Number_of_Components = 2                                           #specify number of components
Initial_Mean_Guess   = np.random.uniform(np.min(Data),np.max(Data), Number_of_Components)
Initial_Covariance_Guess= np.random.uniform(np.min(Data),np.max(Data), Number_of_Components)
Initial_Parameter_Guess= np.random.dirichlet(np.ones(2),size=1)    #must sum to one


# In[15]:


################################
#Seperate Functions
################################
def Normal_Calculation(x, Means, Covariances, Num_Components):
    Normal_Matrix= np.zeros(shape=(np.shape(x)[0],Num_Components)) #initialize and empty vector
    for i in range(Num_Components): #compute the normal probability value for every element in the matrix 
            Normal_Matrix[:,i]=(1/(2*np.pi*Covariances[i])**.5)*np.exp(-np.square(x-Means[i])/(2*Covariances[i]))
    return Normal_Matrix
#For one specific observation: 
def One_Normal_Calculation(x, Mean, Covariance):
    return (1/(2*np.pi*Covariance)**.5)*np.exp(-np.square(x-Mean)/(2*Covariance))


# In[16]:


################################
#Expectation step
#Evaluate the Responsibilities
################################
Normal_Matrix=Normal_Calculation(Data,Initial_Mean_Guess, Initial_Covariance_Guess,Number_of_Components)
Responsibilities_Numerator_Mat = Initial_Parameter_Guess*Normal_Matrix
Responsibilities_Denominator_Mat= np.sum(Initial_Parameter_Guess*Normal_Matrix,axis=1)
Responsibilities=np.zeros(shape=(np.shape(Normal_Matrix)[0],Number_of_Components))
for i in range(np.shape(Normal_Matrix)[0]):
    Responsibilities[i,:]=Responsibilities_Numerator_Mat[i,:]/Responsibilities_Denominator_Mat[i]


# In[17]:


################################
#Maximization Step 
#Re-estimate the parameters using the new responsibilites
################################
New_Mu= np.dot(Responsibilities.T,Data)/ np.sum(Responsibilities,axis=0)
New_Covariance= np.sum(Responsibilities*np.array([np.square(Data-New_Mu[i]) for i in range(Number_of_Components)]).T,
                       axis=0)/np.sum(Responsibilities,axis=0)
New_Parameters= (1/np.shape(Data)[0])*np.sum(Responsibilities,axis=0)


# In[18]:


################################
#Likelihood Step
#Evaluate the Initial Log-Likelihood function
################################
log_likelihood= np.sum(np.log(np.sum(New_Parameters*Normal_Matrix,axis=1)))


# In[19]:


################################
#E-M Iteration step
#Run the expectation-maximization algorithm until the log-likelihood function converges
################################
#num_iters=1000
Tolerance=.0000001
likelihood_difference= Tolerance +1 #initialize to something greater than tolerance
i=1
#for i in range(num_iters):
while likelihood_difference >= Tolerance:
    #Compute Responsibilities
    Normal_Matrix=Normal_Calculation(Data,New_Mu, New_Covariance,Number_of_Components)
    Responsibilities_Numerator_Mat = New_Parameters*Normal_Matrix
    Responsibilities_Denominator_Mat= np.sum(New_Parameters*Normal_Matrix,axis=1)
    Responsibilities=np.zeros(shape=(np.shape(Normal_Matrix)[0],Number_of_Components))
    for j in range(np.shape(Normal_Matrix)[0]):
        Responsibilities[j,:]=Responsibilities_Numerator_Mat[j,:]/Responsibilities_Denominator_Mat[j] 
    #compute new parameters
    New_Mu= np.dot(Responsibilities.T,Data)/ np.sum(Responsibilities,axis=0)
    New_Covariance= np.sum(Responsibilities*np.array([np.square(Data-New_Mu[i]) for i in range(Number_of_Components)]).T,
                           axis=0)/np.sum(Responsibilities,axis=0)
    New_Parameters= (1/np.shape(Data)[0])*np.sum(Responsibilities,axis=0)
    new_log_likelihood=np.sum(np.log(np.sum(New_Parameters*Normal_Matrix,axis=1))) #compute new likelihood
    likelihood_difference= new_log_likelihood-log_likelihood                       #compute the difference
    log_likelihood=new_log_likelihood                                              #update the old likelihood to the new one
    print('iter:', i)
    print('New_Mu:', New_Mu)
    print('New_Covariance', np.sqrt(New_Covariance))
    print('New_Parameters', New_Parameters)
    print('Likelihood:', new_log_likelihood)
    print('Likelihood Difference:', likelihood_difference)
    i+=1


# In[20]:


######################
#Visualization Block
#Visualize the fit of the Gaussian Mixture Model
######################
GMM= np.zeros([len(Data),])
for k in range(Number_of_Components):
    GMM = GMM + New_Parameters[k]*One_Normal_Calculation(Data,New_Mu[k],New_Covariance[k])


# In[21]:


plt.hist(Data,bins='auto', density=True, alpha=.5 , edgecolor='black', color='lightblue')
plt.scatter(Data,GMM,color='red', marker="_", s=2)
plt.title('Data Overlaid with Gaussian Mixture Model')
plt.xlabel('Data')
plt.ylabel('Density')
red_patch = mpatches.Patch(color='red', label='Gaussian Mixture Model')
blue_patch = mpatches.Patch(color='lightblue', label='Observed Data')
plt.legend(handles=[red_patch,blue_patch])
plt.show()

