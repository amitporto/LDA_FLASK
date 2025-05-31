from flask import Flask, render_template, request, Response
import pandas as pd
import csv
from applicability import apdom
import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score,matthews_corrcoef,f1_score,recall_score,confusion_matrix,roc_auc_score, roc_curve, auc
from statsmodels.multivariate.manova import MANOVA


app=Flask(__name__)

app.config['SECRET_KEY']='AmitMLR'

lda=LinearDiscriminantAnalysis(solver='lsqr')

def stat_params(yobs,ypred):
    cm=confusion_matrix(yobs,ypred)
    acc=round(accuracy_score(yobs,ypred),3)
    mcc=round(matthews_corrcoef(yobs,ypred),3)
    f1=round(f1_score(yobs,ypred),3)
    rc=round(recall_score(yobs,ypred),3)
    roc=round(roc_auc_score(yobs,ypred),3)
    tp,tn,fp,fn=cm.ravel()[3],cm.ravel()[0],cm.ravel()[1],cm.ravel()[2]
    return tp,tn,fp,fn,acc,mcc,f1,rc,roc

def fit_linear_reg(X,y):
    dp=pd.concat([X,y],axis=1)
    table=MANOVA.from_formula('X.values~ y.values', data=dp).mv_test().results['y.values']['stat']
    Wilks_lambda=table.iloc[0,0]
    F_value=table.iloc[0,3]
    p_value=table.iloc[0,4]
    return round(Wilks_lambda,3),round(F_value,3),round(p_value,3)

def corrl(df):
    lt=[]
    df1=df.iloc[:,0:]
    for i in range(len(df1)):
        x=df1.values[i]
        x = sorted(x)[0:-1]
        lt.append(x)
    flat_list = [item for sublist in lt for item in sublist]
    return max(flat_list),min(flat_list)

def corr_plot(X_train):
    sb.set(font_scale=2.0)
    corr=pd.DataFrame(X_train).corr()
    fig=plt.figure(figsize=(18,12))
    mask = np.triu(np.ones_like(pd.DataFrame(X_train).corr()))
    dataplot = sb.heatmap(pd.DataFrame(X_train).corr(), cmap="YlGnBu", annot=True, mask=mask)
    plt.tick_params(labelsize=18)
    return fig

def plot_roc_curves(train_df, test_df, true_col, pred_col, plot_title="ROC Curves for Training and Test Sets"):
    """
    Generates ROC curves for training and test sets from experimental and predicted values.

    Parameters:
    train_df (pd.DataFrame): The DataFrame containing the training data.
    test_df (pd.DataFrame): The DataFrame containing the test data.
    true_col (str): The column name for true activity values (binary: 0 or 1).
    pred_col (str): The column name for predicted probabilities.
    plot_title (str): Title of the plot.

    Returns:
    None
    """
    # Extract true labels and predicted probabilities for training set
    y_train_true = train_df[true_col]
    y_train_pred = train_df[pred_col]

    # Compute ROC curve and AUC for training set
    fpr_train, tpr_train, _ = roc_curve(y_train_true, y_train_pred)
    roc_auc_train = auc(fpr_train, tpr_train)

    # Extract true labels and predicted probabilities for test set
    y_test_true = test_df[true_col]
    y_test_pred = test_df[pred_col]

    # Compute ROC curve and AUC for test set
    fpr_test, tpr_test, _ = roc_curve(y_test_true, y_test_pred)
    roc_auc_test = auc(fpr_test, tpr_test)

    # Plot ROC curves
    fig=plt.figure(figsize=(10, 8))
    plt.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Training set ROC (AUC = {roc_auc_train:.2f})')
    plt.plot(fpr_test, tpr_test, color='green', lw=2, label=f'Test set ROC (AUC = {roc_auc_test:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title(plot_title, fontsize=13)
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(alpha=0.3)
    return fig

def fig_standPlot(x,y):
    fig = plt.figure(figsize = (20, 10))
    plt.bar(x,y,align='center') # A bar chart
    plt.xlabel('Descriptors', fontsize = 28)
    plt.ylabel('Standardized coefficients', fontsize = 28)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    return fig

def make_activity_plot(y_true: np.array,
                       y_pred: np.array,
                       y_testtrue: np.array,
                       y_testpred: np.array,
                       xLabel: str = 'True values',
                       yLabel: str = 'Predicted values',
                       r2Score: float = None,
                       rmse: float = None,
                       ):

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.scatter(y_true, y_pred)
    ax.scatter(y_testtrue, y_testpred)
    ax.legend(['Training','Test'],loc='upper left', fontsize=15)
    #ax.set_xlabel('Train')
    # define axis_limits
    #xLabel='Train'
    #yLabel='Test'
    high_activity_lim = np.ceil(max(max(y_true), max(y_pred)))
    low_activity_lim = np.floor(min(min(y_true), min(y_pred)))
    limits = (high_activity_lim, low_activity_lim)

    # display metrics if available
    r2Score = r2Score if r2Score is not None else ''
    rmse = rmse if rmse is not None else ''
    high_annotate_lim = high_activity_lim - 0.2
    low_annotate_lim = low_activity_lim + 0.2
    #ax.annotate("R-squared = {}\nRMSE = {}".format(r2Score, rmse), (low_annotate_lim, high_annotate_lim))
    
    # add regression line
    m, b = np.polyfit(y_true.flatten(), y_pred.flatten(), 1)
    x = np.arange(low_activity_lim, high_activity_lim+1)
    ax.plot(x, m*x+b, color='r')

    # set limits on plot
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.set_xlabel(xLabel, fontsize=20)
    ax.set_ylabel(yLabel, fontsize=20)

    return fig, ax

def sklearn_vif(exogs, data):

    # initialize dictionaries
    vif_dict, tolerance_dict = {}, {}

    # form input data for each exogenous variable
    for exog in exogs:
        not_exog = [i for i in exogs if i != exog]
        X, y = data[not_exog], data[exog]

        # extract r-squared from the fit
        r_squared = LinearRegression().fit(X, y).score(X, y)

        # calculate VIF
        vif = 1/(1 - r_squared)
        vif_dict[exog] = vif

        # calculate tolerance
        tolerance = 1 - r_squared
        tolerance_dict[exog] = tolerance

    # return VIF DataFrame
    df_vif = pd.DataFrame({'VIF': vif_dict, 'Tolerance': tolerance_dict})

    return df_vif


@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')


@app.route('/data', methods=['GET','POST'])
def data():
    if request.method == 'POST':
       file_tr = request.form['csvfile_tr']
       data_tr = pd.read_csv(file_tr)
       file_ts = request.form['csvfile_ts']
       data_ts = pd.read_csv(file_ts)
       ntr=data_tr.iloc[:,0:1]
       nts=data_ts.iloc[:,0:1]
       if request.form['options']=='first':
          Xtr=data_tr.iloc[:,2:]
          ytr=data_tr.iloc[:,1:2]
       elif request.form['options']=='last':
          Xtr=data_tr.iloc[:,1:-1]
          ytr=data_tr.iloc[:,-1:]
       Xts=data_ts[Xtr.columns]
       yts=data_ts[ytr.columns]
       ytr.columns=['Active']
       yts.columns=['Active']

       #Correlation plot
       global dc
       dc=Xtr.corr()
       mx,mn=corrl(dc)
       mc=max(abs(mx),abs(mn))
       global fig_corr
       fig_corr=corr_plot(Xtr)
       #End

       #LDA model fitting and parameters
       lda.fit(Xtr,ytr)
       ytrpr=pd.DataFrame(lda.predict(Xtr),columns=['Predict'])
       ytrpr2=pd.DataFrame(lda.predict_proba(Xtr))
       #ytrpr2.columns=['%Prob(-1)','%Prob(+1)']
       ytrpr2.columns=['prob1','prob2']
       ytrpr2['%Prob(-1)']=ytrpr2.apply(lambda x: round(x['prob1'],3), axis=1)
       ytrpr2['%Prob(+1)']=ytrpr2.apply(lambda x: round(x['prob2'],3), axis=1)
       ytrpr2=ytrpr2.drop(['prob1','prob2'], axis=1)
       ytspr=pd.DataFrame(lda.predict(Xts), columns=['Predict'])
       ytspr2=pd.DataFrame(lda.predict_proba(Xts))
       #ytspr2.columns=['%Prob(-1)','%Prob(+1)']
       ytspr2.columns=['prob1','prob2'] 
       ytspr2['%Prob(-1)']=ytspr2.apply(lambda x: round(x['prob1'],3), axis=1)
       ytspr2['%Prob(+1)']=ytspr2.apply(lambda x: round(x['prob2'],3), axis=1)
       ytspr2=ytspr2.drop(['prob1','prob2'], axis=1)      
       stds = Xtr.std(axis=0, ddof=0).values
       raw_coeffs = lda.coef_[0]
       standardized_coeffs =raw_coeffs * stds
       sc=[round(i,3) for i in standardized_coeffs]
       lx=[round(i,3) for i in lda.coef_[0]]
       ln='Activity = '
       for i,j in zip(lx,list(Xtr)):
           ln=ln+str(i)+'*'+str(j)+'+'
           eq=ln+str(round(lda.intercept_[0],3))  
       Wilks_lambda,F_value,p_value=fit_linear_reg(Xtr,ytr)   
       tptr,tntr,fptr,fntr,acctr,mcctr,f1tr,rctr,roctr=stat_params(ytr,ytrpr)
       tpts,tnts,fpts,fnts,accts,mccts,f1ts,rcts,rocts=stat_params(yts,ytspr)
       #End

       #for raw and standardized coefficients
       desls=list(Xtr)+['Constant']
       deslx=lx+[round(lda.intercept_[0],3)]
       desTable=pd.concat([pd.DataFrame(desls),pd.DataFrame(deslx)], axis=1)
       desTable.columns=['Descriptor','Coefficient']
       desls=list(Xtr)+['Constant']
       deslx=lx+[round(lda.intercept_[0],3)]
       scx=sc+[0.000]
       desTable=pd.concat([pd.DataFrame(desls),pd.DataFrame(deslx),pd.DataFrame(scx)], axis=1)
       desTable.columns=['Descriptor','Raw Coefficient', 'Standardized Coefficient']
       x=desTable.iloc[0:-1,:]['Descriptor']
       y=desTable.iloc[0:-1,:]['Standardized Coefficient']
       global fig_desc
       fig_desc=fig_standPlot(x,y)
       #End
        
       #Applicability domain
       adstr=apdom(Xtr,Xtr)
       yadstr=adstr.fit()
       global ftr
       ftr=pd.concat([ntr,Xtr,ytr,ytrpr,ytrpr2,yadstr],axis=1)
       adsts=apdom(Xts,Xtr)
       yadsts=adsts.fit()
       #global dfts
       global fts
       fts=pd.concat([nts,Xts,yts,ytspr, ytspr2,yadsts],axis=1)
       #End

       global fig_roc
       fig_roc=plot_roc_curves(ftr, fts, 'Active', 'Predict', plot_title="ROC Curves for Training and Test Sets")
       
       return render_template('data.html', tptr=tptr,tntr=tntr, fptr=fptr,fntr=fntr,acctr=acctr,
              mcctr=mcctr,f1tr=f1tr,rctr=rctr, roctr=roctr,tpts=tpts,
              tnts=tnts, fpts=fpts, fnts=fnts,accts=accts, mccts=mccts, f1ts=f1ts,rcts=rcts,rocts=rocts,
              trsize=ntr.shape[0],tssize=nts.shape[0], mc=mc,tbl=desTable.to_html(),eq=eq)

@app.route('/resultsTR')
def results_tr():
    #return redirect(url_for("data", category=category, _external=True, _scheme='https'))
    return render_template('results.html', result =ftr.to_html(index=False))

@app.route('/resultsTS')
def results_ts():
    #return redirect(url_for("data", category=category, _external=True, _scheme='https'))
    return render_template('results.html', result =fts.to_html(index=False))

@app.route('/correlmatrix')
def correlmatrix():
    #return redirect(url_for("data", category=category, _external=True, _scheme='https'))
    return render_template('results.html', result =dc.to_html())

@app.route('/vif')
def vif():
    #return redirect(url_for("data", category=category, _external=True, _scheme='https'))
    return render_template('results.html', result =vif.to_html(index=False))


@app.route('/plot_corr.png')
def plot_png():
    output = io.BytesIO()
    FigureCanvas(fig_corr).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_roc.png')
def plot_png3():
    output = io.BytesIO()
    FigureCanvas(fig_roc).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_desc.png')
def plot_png4():
    output = io.BytesIO()
    FigureCanvas(fig_desc).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


if __name__=='__main__':
  app.run(debug=True)
