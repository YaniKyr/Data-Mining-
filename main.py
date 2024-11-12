import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn import metrics  


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import tensorflow as tf

import dataArchive as dA
from statsmodels.graphics.tsaplots import plot_pacf


from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
from feature_engine.discretisation import EqualFrequencyDiscretiser 
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator, BicScore
import networkx as nx
import pgmpy.metrics as pgmm
import time



#Load multiple cvs files in a dataframe 
def load_data():
    data = pd.DataFrame()
    for _, value in dA.csvs.items():
        temp_data = pd.read_csv(value)
        data = pd.concat([data, temp_data])
    return data



def make_lags(data, lags):
    return pd.concat([data, pd.DataFrame(
        {
            f'{col}_lag_{i}': data[col].shift(i)
            for col in data.columns
            for i in range(1, lags + 1)
        })],
        axis=1)



def dataframePreparation():
    data = load_data()
   
    #data = data.drop(data.columns[-1:], axis=1)
    data = data.dropna()

    #Summary of the dataset
    print(data.describe())

    #Remove timestamp from the dataset
    data = data.drop(columns=['timestamp'])
    
    
    
    #transform the data with z transformation
    scaler = StandardScaler()
    data[dA.sensors] = scaler.fit_transform(data[dA.sensors])


    # Plot partial autocorrelation for each sensor
    #for sensor in dA.sensors:
    #    plt.figure(figsize=(10, 5))
    #    plot_pacf(data[sensor])
    #    plt.title(f'Partial Autocorrelation for {sensor}')
    #    plt.xlabel('Lags')
    #    plt.ylabel('Partial Autocorrelation')
    #    plt.savefig(f'results/pacf_{sensor}.png')
    #    
    
    #Split data into train 90% and test 10% and also labels
    #add lag to all columns of 60
    print("Adding lag to the dataset...")
 
 

    
    lag = 5
    data = make_lags(data, lag)
    data.fillna(0, inplace=True)

    print("Data preparation...")
    train, test = train_test_split(data, test_size=0.2)
    
    print(data.describe())
    
    labels_train = train['label']
    labels_test = test['label']
    #replace values with the index of the activity
    
    #one hot encoding labels
    #get all unique values from dataframe
    
    encoder = OneHotEncoder()
    #labels_train = encoder.fit_transform(labels_train.values.reshape(-1, 1)).toarray()
    #labels_test = encoder.transform(labels_test.values.reshape(-1, 1)).toarray()



    train = train.drop(columns=train.filter(regex='label').columns)
    test = test.drop(columns=test.filter(regex='label').columns)
    
    #from dictionary to numpy array
    #train = train.to_numpy()
    #test = test.to_numpy()
   #
    #train = train.reshape(train.shape[0],len(dA.sensors),lag+1)
    #
    #test = test.reshape(test.shape[0],len(dA.sensors),lag+1)
    #
    #runNNmodel(train, test, labels_train, labels_test)

    runRandomForest(train, test, labels_train, labels_test)
    data = pd.concat([data['label'],data.drop(columns=data.filter(regex='label').columns)],axis=1)
    
    #runBayesianNetwork(data)

def classificationResults(test,pred,save=False,filepath=None,classifier=None):
    
    print("Printing classification report...")
    report = metrics.classification_report( test, pred)
    cm = metrics.confusion_matrix(test, pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d",cmap="Blues")
    plt.xticks(rotation=45)
    if save:
        plt.savefig(f'{filepath}/{classifier}/cm.png')
        #save report to file
        with open(f'{filepath}/{classifier}/report.txt', 'w') as f:
            f.write(report)
    else:
        print(report)    
        plt.show()



def runNNmodel(train, test, labels_train, labels_test):
    import tensorflow as tf
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM( 32, input_shape=(train.shape[1], train.shape[2])),
        tf.keras.layers.Dense(labels_train.shape[1], activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy','f1_score','mean_squared_error','categorical_crossentropy'])
    history = model.fit(train, labels_train, epochs=10, batch_size=64)
    model.evaluate(test, labels_test, verbose=2)
    predictions = model.predict(test)
    
    classificationResults(np.argmax(labels_test,axis=1),np.argmax(predictions,axis=1),save=True,filepath='results',classifier='neural_network')
    

#RandomForest
def runRandomForest(train, test, labels_train, labels_test):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=30, random_state=42)
    print("Training the model...")
    clf.fit(train, labels_train,)
    print("Making predictions...")
    predictions = clf.predict(test)

    # using metrics module for accuracy calculation
    #print("ACCURACY OF THE MODEL:", metrics.accuracy_score(labels_test, predictions))
    
    #call plot confusion matrix
    classificationResults(labels_test,predictions,filepath='results',classifier='random_forest')
    
#Create bayesian network classifier with pgmpy
def BNevaulate(model,test):
    infer = VariableElimination(model)
    y_pred_bn =[]
    counter = 0
    num_rows = len(test)
    #calculate the time elapsed
    start_time = time.time()
    for _, row in test.drop('label', axis=1).iterrows():
        y_pred_bn.append(infer.map_query(variables=['label'], evidence=row.to_dict(), show_progress = False)['label']) 
        counter += 1
        if counter % 1000 == 0:
            elapsed_time = time.time() - start_time
            print(f"Iter: {counter}/{num_rows}, Time elapsed: {elapsed_time:.2f} seconds")
    accuracy_bn = metrics.accuracy_score(test['label'], y_pred_bn)
    print("Accuracy of Bayesian Network:", accuracy_bn)
    return y_pred_bn

def modelSelection(train, selector='hc'):
    match selector:
        case 'hc':
            hc = HillClimbSearch(train)
            best_model = hc.estimate(scoring_method=BicScore(train))
            edges = list(best_model.edges())
            return BayesianNetwork(edges)
        
        case 'star':
            edges = [('back_x', 'label'), ('back_y', 'label'), ('back_z', 'label'), ('thigh_x', 'label'), ('thigh_y', 'label'), ('thigh_z', 'label')]
            return BayesianNetwork(edges)
    
    
        case 'corr':
            
            edges = [('back_x', 'label'), ('thigh_x','thigh_z'),('thigh_x','thigh_y'),
                    ('thigh_x','back_x'),('thigh_x','label'),('back_y','label'),
                    ('back_y','thigh_y'),('back_y','back_z'),('thigh_z','label'),('thigh_y','label'),('back_z','label')]
            return BayesianNetwork(edges)


def plotBN(model,save=False,filepath=None,classifier=None):
    pos = nx.circular_layout(model)
    nx.draw(model, pos=pos, with_labels=True)
    if save:
        
        plt.savefig(f'{filepath}/{classifier}/network.png')
    plt.clf()  # Clear the current figure to prevent overlapping    
def runBayesianNetwork(data):
    columns = [i for i in data.columns[1:]]
    num_bins = 3
    
    disc = EqualFrequencyDiscretiser(q=num_bins, variables=columns)
    
    disc.fit(data)
    equalfrequency_discretizer_dict = disc.binner_dict_
    bin_df_equalfrequency = pd.DataFrame.from_dict(equalfrequency_discretizer_dict, orient = 'index') 
    
    
    print(bin_df_equalfrequency)

    #create ranges of values
    for sensor in columns:
        descritized = [f'{"{:1.2f}".format(min(data[sensor].unique()))} to {"{:1.2f}".format(bin_df_equalfrequency[1][sensor])}']
        for i in range(2,num_bins):
            descritized.append(f'{"{:1.2f}".format(bin_df_equalfrequency[i-1][sensor])} to {"{:1.2f}".format(bin_df_equalfrequency[i][sensor])}')
    
        descritized.append(f'{"{:1.2f}".format(bin_df_equalfrequency[num_bins-1][sensor])} to {"{:1.2f}".format(max(data[sensor].unique()))}')

        data[sensor] = pd.cut(data[sensor], bins=num_bins, labels=descritized)
        
    train, test = train_test_split(data, test_size=0.1)

    

    model = modelSelection(train, selector='hc')

    
    


    model.fit(train,estimator = MaximumLikelihoodEstimator)

    y_pred = BNevaulate(model,test)

   
    classificationResults(test['label'],y_pred,save=True,filepath='results',classifier=f'bayesian_network_hc')
    
    plotBN(model,save=True,filepath='results',classifier=f'bayesian_network_hc')


dataframePreparation()