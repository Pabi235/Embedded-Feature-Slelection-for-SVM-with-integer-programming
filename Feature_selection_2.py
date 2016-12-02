import pandas as dataview
import numpy as np
import os
from gurobipy import *  #Great software for non linear optimisation problems
                        # need a commercial license, unless you're an academic then they're free!
                        #gurobi.org for implementation and installation details
                        #they prefer anaconda which I happen to be using.



#Import dataset into enviroment



load_data = dataview.read_csv(r'C:\Users\Student.SCISTAT004\Downloads\train.csv')
n_load ,p_load=load_data.shape

n_train=int(n_load*0.6)
n_valid=int(n_load*0.2)
n_test=int(n_load*0.2)

dataset=load_data.iloc[:n_train,:]
valid_data=load_data.iloc[n_train:n_train+n_valid,:]
test_data=load_data.iloc[n_train+n_valid:n_train+n_valid+n_test,:]



#Number of observations for each label in the data set
n_0=dataview.value_counts(dataset['label'].values, sort=False)[0]
n_1=dataview.value_counts(dataset['label'].values, sort=False)[1]
n_2=dataview.value_counts(dataset['label'].values, sort=False)[2]
n_3=dataview.value_counts(dataset['label'].values, sort=False)[3]
n_4=dataview.value_counts(dataset['label'].values, sort=False)[4]
n_5=dataview.value_counts(dataset['label'].values, sort=False)[5]
n_6=dataview.value_counts(dataset['label'].values, sort=False)[6]
n_7=dataview.value_counts(dataset['label'].values, sort=False)[7]
n_8=dataview.value_counts(dataset['label'].values, sort=False)[8]
n_9=dataview.value_counts(dataset['label'].values, sort=False)[9]



#the class takes in as parameters: C:the penalisation parameter, a dataset and integer required for classification
class SVM(object):


    def __init__(self, obs_data, class_int, c):
            self.c = float(c)     # c is the penalty on misclassifications of the model
            self.obs_data=obs_data
            self.class_int = class_int
            self.n_features=784
            self.data_prep()



    # this function converts the class labels to -1 or 1
    # if the observation x_i has required label change the label to +1
    # Make the all other label values -1
    #create a random subset for the "rest class" following the one vs all procedure
    #this ensures balanced classes when feature selection is performed.

    def data_prep(self):
        count = 0
        while count < 10 :
            if count != self.class_int:
                self.obs_data['label'] = self.obs_data['label'].replace(to_replace=count,value=-1)
            count += 1

        self.obs_data['label'] = self.obs_data['label'].replace(to_replace=self.class_int,value=1)
        self.rest_sample=self.obs_data[self.obs_data['label'] == -1].copy()
        self.one_sample=self.obs_data[self.obs_data['label'] == 1].copy()
        self.random_rest_1=self.rest_sample.sample(n=eval("n_{}".format(self.class_int))).copy()#avoiding class imbalance
        self.onevsranrest_1=dataview.concat([self.one_sample,self.random_rest_1])
        self.n_rand,self.nvars=self.onevsranrest_1.shape
        self.labels_train=self.onevsranrest_1['label']
        self.random_train=self.onevsranrest_1.loc[:, 'pixel0':'pixel783']


    #Split up into training set and test set.
    # data_frame.loc[x_start:x_end,y_start:y:_end]
    # using 60% of the data set for training
    # 20% of the data set for parameter estimdation and validation
    # 20% of the data set for testing


    #Next we set up optimisation problem for SVM
    def svm_train(self):
         #Optimise Quadratic Programming problem for SVM
         #This the Mixed Integer Quadratic Programming SVM.
        self.svm_model=Model("SVM_Model")
        #Set up relevant variables
        #beta_i


        self.beta = []
        for count in range(self.n_features):
            
            self.beta.append(self.svm_model.addVar(vtype=GRB.CONTINUOUS,name="beta_{}".format(count)))

        #slack variable xi_i for primal
        self.xi = []
        for count in range(self.n_rand):
            
            self.xi.append(self.svm_model.addVar(vtype=GRB.CONTINUOUS,name="xi_{}".format(count)))
    #binary z_i
        self.z = []
        for count in range(self.n_features):
            self.z.append(self.svm_model.addVar(vtype=GRB.BINARY,name="z_{}".format(count)))

        #b variable:
        b=self.svm_model.addVar(vtype=GRB.CONTINUOUS,name="b")

        self.svm_model.update()
        #We are solving the primal of the SVM optimisation problem.
        #we are solving the following optimisation problem
        #maximise w.r.t beta_vector {lp^2(beta)}+C* sum_i=1^N xi_i
        self.obj= self.c * quicksum(self.xi[k] for k in range(self.n_rand))+ (1/2)*quicksum(self.beta[i]*self.beta[i]
                                                               for i in range(self.n_features))  #try n_features otherwise
        self.svm_model.setObjective(self.obj, sense=GRB.MINIMIZE)
        #SVM primal and MIQP constaints

        #y_i(x^T %% beta +b)+xi_i-1 >=0
        for k in range(self.n_rand):
            self.svm_model.addConstr(self.labels_train.iloc[k] * quicksum(self.beta[g]*self.random_train.iloc[k,g] for g in range (self.n_features))+self.labels_train.iloc[k]*b+self.xi[k]-1,
                           GRB.GREATER_EQUAL,0,name="complementslack_{}".format(k))
       # beta_i =< M * z_i
       #-beta =<M*z_i
       #choose M= 1000 for now change to arbitary M later

        for k in range(self.n_features):
            self.svm_model.addConstr(self.beta[k]-1000*self.z[k],GRB.LESS_EQUAL,0,name="magnitudebeta_{}".format(k))
        for k in range (self.n_features):
            self.svm_model.addConstr(self.beta[k]+1000*self.z[k],GRB.GREATER_EQUAL,0,name="magnitudebeta2_{}".format(k))

         #bound number of features choosen by pareto principe.
         #equivalent to resitricted combinatorial search
         #exponential time compelxity
        self.svm_model.addConstr(quicksum(self.z[k] for k in range(self.n_features)),GRB.EQUAL,int(self.n_features*0.25))
        self.svm_model.update()
        self.svm_model.optimize()
         #if optimal solution exists return the corresponding z vector
         # otherwise return a vector of ones i.e. use all the features.
        self. training_time=self.svm_model.Runtime
        if self.svm_model.status == GRB.Status.OPTIMAL:
            self.z_resultant=[]
            self.betas=[]
            for k in range(self.n_features):
                beta_i = self.svm_model.getVarByName("beta_{}".format(k))
                z_i = self.svm_model.getVarByName("z_{}".format(k))
                value_z=z_i.x
                value_beta=beta_i.x
                self.z_resultant.append(value_z)
                self.betas.append(value_beta)

            return  [self.svm_model.Runtime]  #wana get an idea of the complexity of training thios bad bwoii


#The rest is just implementing it and stuff.
#I didnt know about dictionarys at the time(life would involved so much less typing :/)
#I decided to prechose a range of c values and decided c=50,100,250,500,1000 all to be tested on the validation set.
#they're pretty arb values but turned to be quite a comprehensive range )


#zresults:c=250
#Feature selection for Integer =0
svm_int0c250=SVM(obs_data=dataset.copy(deep=True),class_int=0,c=250)
z_svm0c250=np.asarray(svm_int0c250.svm_train())

#Feature selection for Integer =1
svm_int1c250=SVM(obs_data=dataset.copy(deep=True),class_int=1,c=250)
z_svm1c250=np.asarray(svm_int1c250.svm_train())

#Feature selection for Integer =2
svm_int2c250=SVM(obs_data=dataset.copy(deep=True),class_int=2,c=250)
z_svm2c250=np.asarray(svm_int2c250.svm_train())

#Feature selection for Integer =3
svm_int3c250=SVM(obs_data=dataset.copy(deep=True),class_int=3,c=250)
z_svm3c250=np.asarray(svm_int3c250.svm_train())

#Feature selection for Integer =4
svm_int4c250=SVM(obs_data=dataset.copy(deep=True),class_int=4,c=250)
z_svm4c250=np.asarray(svm_int4c250.svm_train())

#Feature selection for Integer =5
svm_int5c250=SVM(obs_data=dataset.copy(deep=True),class_int=5,c=250)
z_svm5c250=np.asarray(svm_int5c250.svm_train())

#Feature selection for Integer =6
svm_int6c250=SVM(obs_data=dataset.copy(deep=True),class_int=6,c=250)
z_svm6c250=np.asarray(svm_int6c250.svm_train())

#Feature selection for Integer =7
svm_int7c250=SVM(obs_data=dataset.copy(deep=True),class_int=7,c=250)
z_svm7c250=np.asarray(svm_int7c250.svm_train())

#Feature selection for Integer =8
svm_int8c250=SVM(obs_data=dataset.copy(deep=True),class_int=8,c=250)
z_svm8c250=np.asarray(svm_int8c250.svm_train())

#Feature selection for Integer =9
svm_int9c250=SVM(obs_data=dataset.copy(deep=True),class_int=9,c=250)
z_svm9c250=np.asarray(svm_int9c250.svm_train())   





#zresults:c=100
#Feature selection for Integer =0
svm_int0c100=SVM(obs_data=dataset.copy(deep=True),class_int=0,c=100)
z_svm0c100=np.asarray(svm_int0c100.svm_train())
        
#Feature selection for Integer =1
svm_int1c100=SVM(obs_data=dataset.copy(deep=True),class_int=1,c=100)
z_svm1c100=np.asarray(svm_int1c100.svm_train())
        
#Feature selection for Integer =2
svm_int2c100=SVM(obs_data=dataset.copy(deep=True),class_int=2,c=100)
z_svm2c100=np.asarray(svm_int2c100.svm_train())
        
#Feature selection for Integer =3
svm_int3c100=SVM(obs_data=dataset.copy(deep=True),class_int=3,c=100)
z_svm3c100=np.asarray(svm_int3c100.svm_train())
        
#Feature selection for Integer =4
svm_int4c100=SVM(obs_data=dataset.copy(deep=True),class_int=4,c=100)
z_svm4c100=np.asarray(svm_int4c100.svm_train())
        
#Feature selection for Integer =5
svm_int5c100=SVM(obs_data=dataset.copy(deep=True),class_int=5,c=100)
z_svm5c100=np.asarray(svm_int5c100.svm_train())
        
#Feature selection for Integer =6
svm_int6c100=SVM(obs_data=dataset.copy(deep=True),class_int=6,c=100)
z_svm6c100=np.asarray(svm_int6c100.svm_train())
        
#Feature selection for Integer =7
svm_int7c100=SVM(obs_data=dataset.copy(deep=True),class_int=7,c=100)
z_svm7c100=np.asarray(svm_int7c100.svm_train())
        
#Feature selection for Integer =8
svm_int8c100=SVM(obs_data=dataset.copy(deep=True),class_int=8,c=100)
z_svm8c100=np.asarray(svm_int8c100.svm_train())
        
#Feature selection for Integer =9
svm_int9c100=SVM(obs_data=dataset.copy(deep=True),class_int=9,c=100)
z_svm9c100=np.asarray(svm_int9c100.svm_train())   
    

        
        


#zresults:c=50
#Feature selection for Integer =0
svm_int0=SVM(obs_data=dataset.copy(deep=True),class_int=0,c=50)
z_svm0=np.asarray(svm_int0.svm_train())

#Feature selection for Integer =1
svm_int1=SVM(obs_data=dataset.copy(deep=True),class_int=1,c=50)
z_svm1=np.asarray(svm_int1.svm_train())

#Feature selection for Integer =2
svm_int2=SVM(obs_data=dataset.copy(deep=True),class_int=2,c=50)
z_svm2=np.asarray(svm_int2.svm_train())

#Feature selection for Integer =3
svm_int3=SVM(obs_data=dataset.copy(deep=True),class_int=3,c=50)
z_svm3=np.asarray(svm_int3.svm_train())

#Feature selection for Integer =4
svm_int4=SVM(obs_data=dataset.copy(deep=True),class_int=4,c=50)
z_svm4=np.asarray(svm_int4.svm_train())

#Feature selection for Integer =5
svm_int5=SVM(obs_data=dataset.copy(deep=True),class_int=5,c=50)
z_svm5=np.asarray(svm_int5.svm_train())

#Feature selection for Integer =6
svm_int6=SVM(obs_data=dataset.copy(deep=True),class_int=6,c=50)
z_svm6=np.asarray(svm_int6.svm_train())

#Feature selection for Integer =7
svm_int7=SVM(obs_data=dataset.copy(deep=True),class_int=7,c=50)
z_svm7=np.asarray(svm_int7.svm_train())

#Feature selection for Integer =8
svm_int8=SVM(obs_data=dataset.copy(deep=True),class_int=8,c=50)
z_svm8=np.asarray(svm_int8.svm_train())

#Feature selection for Integer =9
svm_int9=SVM(obs_data=dataset.copy(deep=True),class_int=9,c=50)
z_svm9=np.asarray(svm_int9.svm_train())



#zresults:c=500
#Feature selection for Integer =0
svm_int0c500=SVM(obs_data=dataset.copy(deep=True),class_int=0,c=500)
z_svm0c500=np.asarray(svm_int0c500.svm_train())

#Feature selection for Integer =1
svm_int1c500=SVM(obs_data=dataset.copy(deep=True),class_int=1,c=500)
z_svmc500=np.asarray(svm_int1c500.svm_train())

#Feature selection for Integer =2
svm_int2c500=SVM(obs_data=dataset.copy(deep=True),class_int=2,c=500)
z_svm2c500=np.asarray(svm_int2c500.svm_train())

#Feature selection for Integer =3
svm_int3c500=SVM(obs_data=dataset.copy(deep=True),class_int=3,c=500)
z_svm3c500=np.asarray(svm_int3c500.svm_train())

#Feature selection for Integer =4
svm_int4c500=SVM(obs_data=dataset.copy(deep=True),class_int=4,c=500)
z_svm4c500=np.asarray(svm_int4c500.svm_train())

#Feature selection for Integer =5
svm_int5c500=SVM(obs_data=dataset.copy(deep=True),class_int=5,c=500)
z_svm5c500=np.asarray(svm_int5c500.svm_train())

#Feature selection for Integer =6
svm_int6c500=SVM(obs_data=dataset.copy(deep=True),class_int=6,c=500)
z_svm6c500=np.asarray(svm_int6c500.svm_train())

#Feature selection for Integer =7
svm_int7c500=SVM(obs_data=dataset.copy(deep=True),class_int=7,c=500)
z_svm7c500=np.asarray(svm_int7c500.svm_train())

#Feature selection for Integer =8
svm_int8c500=SVM(obs_data=dataset.copy(deep=True),class_int=8,c=500)
z_svm8c500=np.asarray(svm_int8c500.svm_train())

#Feature selection for Integer =9
svm_int9c500=SVM(obs_data=dataset.copy(deep=True),class_int=9,c=500)
z_svm9c500=np.asarray(svm_int9c500.svm_train())   

#zresults:c=1000
#Feature selection for Integer =0
svm_int0c1000=SVM(obs_data=dataset.copy(deep=True),class_int=0,c=1000)
z_svm0c1000=np.asarray(svm_int0c1000.svm_train())

#Feature selection for Integer =1
svm_int1c1000=SVM(obs_data=dataset.copy(deep=True),class_int=1,c=1000)
z_svmc1000=np.asarray(svm_int1c1000.svm_train())

#Feature selection for Integer =2
svm_int2c1000=SVM(obs_data=dataset.copy(deep=True),class_int=2,c=1000)
z_svm2c1000=np.asarray(svm_int2c1000.svm_train())

#Feature selection for Integer =3
svm_int3c1000=SVM(obs_data=dataset.copy(deep=True),class_int=3,c=1000)
z_svm3c1000=np.asarray(svm_int3c1000.svm_train())

#Feature selection for Integer =4
svm_int4c1000=SVM(obs_data=dataset.copy(deep=True),class_int=4,c=1000)
z_svm4c1000=np.asarray(svm_int4c1000.svm_train())

#Feature selection for Integer =5
svm_int5c1000=SVM(obs_data=dataset.copy(deep=True),class_int=5,c=1000)
z_svm5c1000=np.asarray(svm_int5c1000.svm_train())

#Feature selection for Integer =6
svm_int6c1000=SVM(obs_data=dataset.copy(deep=True),class_int=6,c=1000)
z_svm6c1000=np.asarray(svm_int6c1000.svm_train())

#Feature selection for Integer =7
svm_int7c1000=SVM(obs_data=dataset.copy(deep=True),class_int=7,c=1000)
z_svm7c1000=np.asarray(svm_int7c1000.svm_train())

#Feature selection for Integer =8
svm_int8c1000=SVM(obs_data=dataset.copy(deep=True),class_int=8,c=1000)
z_svm8c1000=np.asarray(svm_int8c1000.svm_train())

#Feature selection for Integer =9
svm_int9c1000=SVM(obs_data=dataset.copy(deep=True),class_int=9,c=1000)
z_svm9c1000=np.asarray(svm_int9c1000.svm_train())   


class prediction(object):
    def __init__(self,beta_matrix,test_set):
        self.beta_matrix=beta_matrix
        self.test_set=test_set
        self.n_obs,self.p=self.test_set.shape
        self.predicted=np.array([])
    def predictor(self):
        self.test_obs=self.test_set.loc[:,'pixel0':'pixel783']
        self.hyp_dist_mat_2=np.transpose(np.dot(self.beta_matrix,self.test_obs.transpose()))
        for k in range(self.n_obs):
            current_row=self.hyp_dist_mat_2[k,:]
        #classify in as the label with the furtherst distance for the hyperplane
            self.predicted=np.append(self.predicted,np.argmax(current_row))
        #check the number of correct classifications
        acutal_labels=test_set['label']
        mis_clas=(n_obs-np.count_nonzero(self.predicted,actual_label.as_matrix()))/n_obs
        
mat=np.matrix([z_svm0,z_svm1,z_svm2,z_svm3,z_svm4,z_svm5,z_svm6,z_svm7,z_svm8,z_svm9])
mat=np.matrix([svm_int0.betas,svm_int1.betas,svm_int2.betas,svm_int3.betas,svm_int4.betas,svm_int5.betas,svm_int6.betas,svm_int7.betas,svm_int8.betas,svm_int9.betas])
np.savetxt(r'C:\Users\Pabi\Desktop\Project\Data Set\zresultsbalancedc50.csv', mat, delimiter=',')
            

    

