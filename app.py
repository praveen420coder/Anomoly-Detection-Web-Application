from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from csv import writer
from csv import DictWriter

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("prediction.html")


@app.route('/predict',methods=['POST','GET'])
def predict():


    sourceID = request.form.get("sourceID")
    sourceAddress = request.form.get("sourceAddress")
    sourceType = request.form.get("sourceType")
    sourceLocation = request.form.get("sourceLocation")
    destinationServiceAddress = request.form.get("destinationServiceAddress")
    destinationServiceType = request.form.get("destinationServiceType")
    destinationLocation = request.form.get("destinationLocation")
    accessedNodeAddress = request.form.get("accessedNodeAddress")
    accessedNodeType = request.form.get("accessedNodeType")
    operation = request.form.get("operation")
    value = request.form.get("value")
    # normality = request.form.get("normality")

    my_list = [sourceID,sourceAddress,sourceType,sourceLocation,destinationServiceAddress,destinationServiceType,destinationLocation,accessedNodeAddress,accessedNodeType,operation,value]
    print(my_list)
    my_array = np.array(my_list)
    columns=['sourceID','sourceAddress','sourceType','sourceLocation','destinationServiceAddress','destinationServiceType','destinationLocation','accessedNodeAddress','accessedNodeType','operation','value']
    df_1 = pd.DataFrame(my_array.reshape(-1, len(my_array)),columns=columns)
    dict={'sourceID':sourceID,'sourceAddress':sourceAddress,'sourceType':sourceType,'sourceLocation':sourceLocation,'destinationServiceAddress':destinationServiceAddress,'destinationServiceType':destinationServiceType,'destinationLocation':destinationLocation,'accessedNodeAddress':accessedNodeAddress,'accessedNodeType':accessedNodeType,'operation':operation,'value':value}

    
        
    with open('mainSimulationAccessTraces.csv', 'a', newline='') as f_object:
        # Pass the CSV  file object to the Dictwriter() function
        # Result - a DictWriter object
        dictwriter_object = DictWriter(f_object, fieldnames=columns)
        # Pass the data in the dictionary as an argument into the writerow() function
        dictwriter_object.writerow(dict)
        # Close the file object
        f_object.close()


    df_data_1 = pd.read_csv("mainSimulationAccessTraces.csv")
    df_data_1['accessedNodeType'] = df_data_1['accessedNodeType'].fillna(value='/Malicious')
    df_data_1.loc[df_data_1.value=='twenty',"value"] = '20.0'
    df_data_1.loc[df_data_1.value=='false',"value"] = '0'
    df_data_1.loc[df_data_1.value=='true',"value"] = '1'
    df_data_1.loc[df_data_1.value=='none',"value"] = '0'
    df_data_1.loc[df_data_1.value=='0',"value"] = '0.0'
    df_data_1['value'] = df_data_1['value'].fillna(value='60.0')
    df_data_1 = df_data_1.drop(df_data_1.index[df_data_1.value.str.contains("org.*")])
    df_data_1.value = df_data_1.value.astype(float)

    df_1 = df_data_1.drop('timestamp',axis=1)
    df_1 = df_data_1.drop('normality',axis=1)
    labelencoder0 = LabelEncoder()
    labelencoder1 = LabelEncoder()
    labelencoder4 = LabelEncoder()
    labelencoder7 = LabelEncoder()
    labelencodery = LabelEncoder()

    X_0 = df_1.iloc[:,0].values
    X_1 = df_1.iloc[:,1].values
    X_4 = df_1.iloc[:,4].values
    X_7 = df_1.iloc[:,7].values
    y = df_1.iloc[:,11].values

    X_0 = labelencoder0.fit_transform(X_0)
    X_0 = X_0.reshape(len(X_0),1)
    X_1 = labelencoder1.fit_transform(X_1)
    X_1 = X_1.reshape(len(X_1),1)
    X_4 = labelencoder4.fit_transform(X_4)
    X_4 = X_4.reshape(len(X_4),1)
    X_7 = labelencoder7.fit_transform(X_7)
    X_7 = X_7.reshape(len(X_7),1)
    y = labelencodery.fit_transform(y)


    ohe_2 = OneHotEncoder()
    ohe_3 = OneHotEncoder()
    ohe_5 = OneHotEncoder()
    ohe_6 = OneHotEncoder()
    ohe_8 = OneHotEncoder()
    ohe_9 = OneHotEncoder()
    le_2 = LabelEncoder()
    le_3 = LabelEncoder()
    le_5 = LabelEncoder()
    le_6 = LabelEncoder()
    le_8 = LabelEncoder()
    le_9 = LabelEncoder()
    X_2 = df_1.iloc[:,2].values
    X_3 = df_1.iloc[:,3].values
    X_5 = df_1.iloc[:,5].values
    X_6 = df_1.iloc[:,6].values
    X_8 = df_1.iloc[:,8].values
    X_9 = df_1.iloc[:,9].values
    X_10 = df_1.iloc[:,10].values

    X_2 = le_2.fit_transform(X_2)
    X_2 = X_2.reshape(len(X_2),1)
    X_3 = le_3.fit_transform(X_3)
    X_3 = X_3.reshape(len(X_3),1)
    X_5 = le_5.fit_transform(X_5)
    X_5 = X_5.reshape(len(X_5),1)
    X_6 = le_6.fit_transform(X_6)
    X_6 = X_6.reshape(len(X_6),1)
    X_8 = le_8.fit_transform(X_8)
    X_8 = X_8.reshape(len(X_8),1)
    X_9 = le_9.fit_transform(X_9)
    X_9 = X_9.reshape(len(X_9),1)
    X_10 = X_10.reshape(len(X_10),1)

    X = np.concatenate((X_0,X_1,X_2,X_3,X_4,X_5,X_6,X_7,X_8,X_9,X_10),axis=1)
    y_resized = y.reshape(len(y),1)
    
    df_spark = pd.DataFrame(X)
    x = df_spark.tail(1)




    prediction = model.predict_proba(x)
    output1='{0:.{1}f}'.format(prediction[0][0], 2)
    output2='{0:.{1}f}'.format(prediction[0][1], 2)
    output3='{0:.{1}f}'.format(prediction[0][2], 2)
    output4='{0:.{1}f}'.format(prediction[0][3], 2)
    output5='{0:.{1}f}'.format(prediction[0][4], 2)
    output6='{0:.{1}f}'.format(prediction[0][5], 2)
    output7='{0:.{1}f}'.format(prediction[0][6], 2)
    output8='{0:.{1}f}'.format(prediction[0][7], 2)

    print(prediction)
    #<!-- <p>['anomalous(DoSattack)', 'anomalous(dataProbing)','anomalous(malitiousControl)', 'anomalous(malitiousOperation)','anomalous(scan)', 'anomalous(spying)', 'anomalous(wrongSetUp)','normal']</p> -->

    if output1>str(0.5):
        return render_template('prediction.html',pred='Your Device is in Danger.\nYour device infected with Dos attack',warning="Do Something ASAP....")
    elif output2>str(0.5):
        return render_template('prediction.html',pred='Your Device is in Danger.\nData Probing Anomaly found',warning="Do Something ASAP....")
    elif output3>str(0.5):
        return render_template('prediction.html',pred='Your Device is in Danger.\nMalitious Control anomaly found',warning="Do Something ASAP....")
    elif output4>str(0.5):
        return render_template('prediction.html',pred='Your Device is in Danger.\nMalitious Operation anomaly found',warning="Do Something ASAP....")
    elif output5>str(0.5):
        return render_template('prediction.html',pred='Your Device is in Danger.\nScan Anomaly found',warning="Do Something ASAP....")
    elif output6>str(0.5):
        return render_template('prediction.html',pred='Your Device is in Danger.\nSomeone is Spying Your Device',warning="Do Something ASAP....")
    elif output7>str(0.5):
        return render_template('prediction.html',pred='Your Device is in Danger.\nWromg Set Up Anomaly Found',warning="Do Something ASAP....")
    else:
        return render_template('prediction.html',pred='Your Device is safe.\n No Anomaly Found',warning="Your Device is Safe for now..")


if __name__ == '__main__':
    app.run(debug=True)
