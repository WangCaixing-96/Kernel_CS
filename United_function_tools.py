import random
import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.metrics import mean_squared_error
from cvxopt import matrix, solvers


def Kernel_rbf(x_1,x_2,sigma=1.0):
    """function to calculate Gaussian kernel"""
    if x_1.ndim==1:
        x_1=x_1.reshape(x_1.shape[0],1)
    if x_2.ndim==1:
        x_2=x_2.reshape(x_2.shape[0],1)
        
    dist_sq=np.sum(x_1**2,1).reshape(-1,1)+np.sum(x_2**2,1)-2*np.dot(x_1,x_2.T)
    K=np.exp(-sigma * dist_sq)
    return K




#Kernel ridge regression (KRR)
def KRR_estimation(x,x_train,y_train,x_valid=None,f_valid=None,weighted=False,weight=None,truncation=False,tr=None,lam=0.3,sigma=1):
    """function to accomplish estimation for kernel ridge regression"""
    n=x_train.shape[0]
    if weighted==False:
        W=np.diag(np.ones(n))
    else:
        W=np.diag(weight)
        
    K=Kernel_rbf(x_train,x_train,sigma=sigma)
    if truncation==False:
        a=np.linalg.inv(np.dot(W,K)+n*lam*np.identity(n))
        b=np.dot(a,W)
        theta_hat=np.dot(b,y_train)
        A=Kernel_rbf(x,x_train)
        result=(theta_hat,np.dot(A,theta_hat))
    else: 
        a_max=np.array(tr)
        loss_value=np.zeros(tr.shape[0])
        for k in range(tr.shape[0]):
            weight_tr=np.clip(weight,a_min=-10000, a_max=a_max[k], out=None)
            W_tr=np.diag(weight_tr)
            a=np.linalg.inv(np.dot(W_tr,K)+n*lam*np.identity(n))
            b=np.dot(a,W_tr)
            theta_hat=np.dot(b,y_train)
            A=Kernel_rbf(x_valid,x_train)
            y_hat=np.dot(A,theta_hat)
            loss_value[k]=mean_squared_error(f_valid,y_hat)
        dic=dict(zip(a_max,loss_value))
        dic=dict(sorted(dic.items(),key=lambda x: x[1],reverse=False))
        a_max_op=list(dic.keys())[0]
        
        weight_tr=np.clip(weight,a_min=-10000, a_max=a_max_op, out=None)
        W_tr=np.diag(weight_tr)
        a=np.linalg.inv(np.dot(W_tr,K)+n*lam*np.identity(n))
        b=np.dot(a,W_tr)
        theta_hat=np.dot(b,y_train)
        A=Kernel_rbf(x,x_train)
        result=(theta_hat,np.dot(A,theta_hat))
    return result


#Kernel quantile regression (KQR)
def KQR_estimation(x,x_train,y_train,x_valid=None,f_valid=None,y_valid=None,weighted=False,
                   weight=None,truncation=False,tr=None,tau=0.5,C=0.15,sigma=1,loss='MSE'):
    """function to accomplish estimation for kernel quantile regression"""
    n=x_train.shape[0]
    if weighted==False:
        weight=np.ones(n)
        
    K=Kernel_rbf(x_train,x_train,sigma=sigma) 
    G=np.vstack((np.identity(n),-np.identity(n)))
    
    P=matrix(K)
    q=matrix(-y_train)
    G=matrix(G)
    A=matrix(np.ones(n),(1,n))
    b=matrix([0.0])
    if truncation==False:
        t_1=C*tau*weight
        t_2=C*(1-tau)*weight
        h=np.vstack((t_1.reshape(n,1),t_2.reshape(n,1)))
        h=matrix(h)
        
        result=solvers.qp(P,q,G,h,A,b)
        alpha_hat=np.array(result['x'])
        K_new=Kernel_rbf(x,x_train,sigma=sigma)
        y_hat=np.dot(K_new,alpha_hat)+result['y']
        a=(alpha_hat,y_hat)
    else:
        a_max=np.array(tr)
        loss_value=np.zeros(a_max.shape[0])
        if loss=='MSE':
            for k in range(a_max.shape[0]):
                weight_tr=np.clip(weight,a_min=-10000, a_max=a_max[k], out=None)
                t_1=C*tau*weight_tr
                t_2=C*(1-tau)*weight_tr
                h=np.vstack((t_1.reshape(n,1),t_2.reshape(n,1)))
                h=matrix(h)
        
                result=solvers.qp(P,q,G,h,A,b)
                alpha_hat=np.array(result['x'])
                K_new=Kernel_rbf(x_valid,x_train,sigma=sigma)
                y_hat=np.dot(K_new,alpha_hat)+result['y']
                loss_value[k]=mean_squared_error(f_valid,y_hat)
        else:
            for k in range(a_max.shape[0]):
                weight_tr=np.clip(weight,a_min=-10000, a_max=a_max[k], out=None)
                t_1=C*tau*weight_tr
                t_2=C*(1-tau)*weight_tr
                h=np.vstack((t_1.reshape(n,1),t_2.reshape(n,1)))
                h=matrix(h)
        
                result=solvers.qp(P,q,G,h,A,b)
                alpha_hat=np.array(result['x'])
                K_new=Kernel_rbf(x_valid,x_train,sigma=sigma)
                y_hat=np.dot(K_new,alpha_hat)+result['y']
                loss_value[k]=empirical_KQR(y=y_valid,f_hat=y_hat,f_true=f_valid,tau=tau)
        dic=dict(zip(a_max,loss_value))
        dic=dict(sorted(dic.items(),key=lambda x: x[1],reverse=False))
        a_max_op=list(dic.keys())[0]
        
        weight_tr=np.clip(weight,a_min=-10000, a_max=a_max_op, out=None)
        t_1=C*tau*weight_tr
        t_2=C*(1-tau)*weight_tr
        h=np.vstack((t_1.reshape(n,1),t_2.reshape(n,1)))
        h=matrix(h)
        
        result=solvers.qp(P,q,G,h,A,b)
        alpha_hat=np.array(result['x'])
        K_new=Kernel_rbf(x,x_train,sigma=sigma)
        y_hat=np.dot(K_new,alpha_hat)+result['y']
        a=(alpha_hat,y_hat,a_max_op)
    return a

def l_tau(x,tau=0.5):
    """function to calculate check loss for KQR"""
    re=np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        if x[i]>=0:
            re[i]=tau*x[i]
        else:
            re[i]=(1-tau)*x[i]
    return re
        
def empirical_KQR(y,f_hat,f_true,tau=0.5):
    """function to calculate empirical excess risk for KQR"""
    y=y.reshape(y.shape[0],)
    f_hat=f_hat.reshape(f_hat.shape[0],)
    f_true=f_true.reshape(f_true.shape[0],)
    a=l_tau(y-f_hat,tau=tau)
    b=l_tau(y-f_true,tau=tau)
    c=np.abs(a-b)
    return np.mean(c)  


#Kernel logistic regression (KLR)
def Newton(x,y,alpha,weight,lamb=0.1,sigma=1):
    """function to perform Newton–Raphson algorithm for KLR"""
    n=y.shape[0]
    K=Kernel_rbf(x,x,sigma=sigma)
    p=np.exp(y*np.dot(K,alpha))/(1+np.exp(y*np.dot(K,alpha)))
    W=np.diag(p*(1-p)*weight)
    
    A=np.linalg.inv(np.dot(W,K)+n*lamb*np.identity(n))
    B=np.dot(np.dot(W,K),alpha)+y*(1-p)*weight
    
    return np.dot(A,B)

def KLR_estimation(x,x_train,y_train,T,x_valid=None,y_valid=None,weighted=False,weight=None,truncation=False,tr=None,lamb=0.1,sigma=1):
    """function to accomplish estimation for kernel logistic regression"""
    n=x_train.shape[0]
    if weighted==False:
        W=np.ones(n)
    else:
        W=weight
    if truncation==False:
        alpha=np.zeros(n)
        for t in range(T):
            alpha=Newton(x=x_train,y=y_train,weight=W,alpha=alpha,lamb=lamb,sigma=sigma)
        K_new=Kernel_rbf(x,x_train,sigma=sigma)
        y_hat=np.dot(K_new,alpha)
        y_pre=np.zeros(x.shape[0])
        y_pre[y_hat>=0]=1
        y_pre[y_hat<0]=-1
        y_pre=y_pre.reshape(x.shape[0],) 
        result=(y_hat,y_pre)
    else:
        a_max=np.array(tr)
        loss_value=np.zeros(tr.shape[0])
        for k in range(tr.shape[0]):
            weight_tr=np.clip(W,a_min=-10000, a_max=a_max[k], out=None)
            alpha=np.zeros(n)
            for t in range(T):
                alpha=Newton(x=x_train,y=y_train,weight=weight_tr,alpha=alpha,lamb=lamb,sigma=sigma)
            K_new=Kernel_rbf(x_valid,x_train,sigma=sigma)
            y_hat=np.dot(K_new,alpha)
            y_pre=np.zeros(x_valid.shape[0])
            y_pre[y_hat>=0]=1
            y_pre[y_hat<0]=-1
            y_pre=y_pre.reshape(x_valid.shape[0],) 
            loss_value[k]=(y_valid!=y_pre).sum()/y_valid.shape[0]
        dic=dict(zip(a_max,loss_value))
        dic=dict(sorted(dic.items(),key=lambda x: x[1],reverse=False))
        a_max_op=list(dic.keys())[0]
        
        weight_tr=np.clip(W,a_min=-10000, a_max=a_max_op, out=None)
        alpha=np.zeros(n)
        for t in range(T):
            alpha=Newton(x=x_train,y=y_train,weight=weight_tr,alpha=alpha,lamb=lamb,sigma=sigma)
        K_new=Kernel_rbf(x,x_train,sigma=sigma)
        y_hat=np.dot(K_new,alpha)
        y_pre=np.zeros(x.shape[0])
        y_pre[y_hat>=0]=1
        y_pre[y_hat<0]=-1
        y_pre=y_pre.reshape(x.shape[0],) 
        result=(y_hat,y_pre)
    return result

def empirical_KLR(y,f_hat,f_true):
    y=y.reshape(y.shape[0],)
    f_hat=f_hat.reshape(f_hat.shape[0],)
    f_true=f_true.reshape(f_true.shape[0],)
    a=np.log(1+np.exp(-y*f_hat))
    b=np.log(1+np.exp(-y*f_true))
    c=np.abs(a-b)
    return np.mean(c)  



#Kernel support vector machine (KSVM)
def KSVM_estimation(x,x_train,y_train,weighted=False,weight=None,C=0.15,sigma=1):
    """function to accomplish estimation for kernel SVM"""
    n_tr=x_train.shape[0]  
    K=Kernel_rbf(x_train,x_train,sigma=sigma)
    K_tilde=np.dot(np.dot(np.diag(y_train),Kernel_rbf(x_train,x_train)),np.diag(y_train))
    G=np.vstack((np.identity(n_tr),-np.identity(n_tr)))
    
    if weighted==False:
        t_1=C*np.ones(n_tr)
        t_2=np.zeros(n_tr)
    else:
        t_1=C*weight
        t_2=np.zeros(n_tr)
        
    h=np.vstack((t_1.reshape(n_tr,1),t_2.reshape(n_tr,1)))
    
    P=matrix(K_tilde)
    q=matrix(-np.ones(n_tr))
    G=matrix(G)
    h=matrix(h)
    A=matrix(y_train,(1,n_tr),'d')
    b=matrix([0.0])
    result=solvers.qp(P,q,G,h,A,b)
    eta_hat=np.array(result['x'])
    alpha_hat=eta_hat*y_train.reshape(n_tr,1)
    K_new=Kernel_rbf(x,x_train,sigma=sigma)
    y_hat=np.dot(K_new,alpha_hat)+result['y']
    
    a=(alpha_hat,y_hat,result['y'])
    return a

def Weighted_loss(y_pre,y_true,weight,ty='one-zero'):
    """function to calculate weighted one-zero or hinge loss for KSVM"""
    if ty=='one-zero':
        loss=((y_pre!=y_true)*weight).mean()
    elif ty=='hinge':
        loss=(np.clip(1-y_pre*y_true, a_min=0,a_max=100000)*weight).mean()
    return loss  
    
    
def cross_validation(x_train,y_train,weight,tr,cv_times=3,random_state=None, ty='one-zero',C=1):
    """function to accomplish cross validation for tuning parameter, say truncation"""
    #split the data
    x_tr_size=x_train.shape[0]
    if random_state is not None:
        np.random.seed(random_state)
        cho=np.random.choice(cv_times,x_tr_size)
    else:
        cho=np.random.choice(cv_times,x_tr_size)
    data=pd.DataFrame(x_train)
    data['y']=y_train
    data['cho']=cho
    data['w']=weight
    loss_value=np.zeros(tr.shape[0])
    for i in range(tr.shape[0]):
        loss_f=np.zeros(cv_times)
        for j in range(cv_times):
            x_tr=np.array(data[data['cho']!=j].iloc[:,:-3])
            y_tr=np.array(data[data['cho']!=j].iloc[:,-3])
            x_te=np.array(data[data['cho']==j].iloc[:,:-3])
            y_te=np.array(data[data['cho']==j].iloc[:,-3])
            weight_tr=np.clip(np.array(data[data['cho']!=j].iloc[:,-1]),a_max=tr[i],a_min=0)
            weight_te=np.array(data[data['cho']==j].iloc[:,-1])
            result=KSVM_estimation(x_te,weight=weight_tr,weighted=True,x_train=x_tr,y_train=y_tr,C=C,sigma=1)
            if ty=='one-zero':
                y_pre=result[1].copy()
                y_pre[y_pre>=0]=1
                y_pre[y_pre<0]=-1
                y_pre=y_pre.reshape(x_te.shape[0],)
                loss_f[j]=Weighted_loss(y_pre=y_pre,y_true=y_te,weight=weight_te,ty='one-zero')
            elif ty=='hinge':
                y_pre=result[1].reshape(x_te.shape[0],)
                loss_f[j]=Weighted_loss(y_pre=y_pre,y_true=y_te,weight=weight_te,ty='hinge')
        loss_value[i]=loss_f.mean()
    dic=dict(zip(tr,loss_value))
    dic=dict(sorted(dic.items(),key=lambda x: x[1],reverse=False))
    tr_op=list(dic.keys())[0]
    return tr_op