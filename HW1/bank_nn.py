import numpy as np   
import time
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import ShuffleSplit
from keras.models import Model
from keras.layers import Input, Dense


#=============GPU enviroment=============
# Use the first GPU card
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Automatically grow GPU memory usage
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Set the session used by Keras
tf.keras.backend.set_session(sess)

#========================================

class PreProcessing():

    def __init__(self, path, train, test):
        self.path = path
        self.traindata = self.path+train
        self.testdata = self.path+test
        
        self.to_int=0
        
    def read_csv(self, f):
        data_list = pd.read_csv(f,sep=",")
        return data_list


    # StandardScaler the data
    def normalized(self, df):
        mean = np.mean(df, axis=0)
        std = np.std(df, axis=0)
        var = std * std

        df_normalized = df - mean
        df_normalized = df_normalized / std

        return df_normalized


    def convert_to_value(self, df):
        # get the value type columns
        value_cols = df.describe().columns
        # get the word type columns
        word_cols = list(set(df.columns)-set(value_cols))

        # make a dictionary to every kind of type
        # ,and change entire word type df to value type
        for col in word_cols:
            get_values = set(df[col].values)
            category_dict = {v:i for i,v in enumerate(get_values)}	        

            arr = []
            for value in df[col].values:
                arr.append(category_dict[value])

            df[col] = np.array(arr)

        return df


    def cross_validation_split(self, df):
        # get the data and output
        X = df.values[:, :-1]
        Y = df.values[:,-1]

        # random split data to 4 pieces, the test size is .25
        # want to use cross validation
        rs = ShuffleSplit(n_splits=4, test_size=.25)
        rs_list = rs.split(X)      

        return X, Y, rs_list


    def label2int(self, ind):
        
        if ind==0:
            self.to_int=0
        elif ind==1:
            self.to_int=1

        return self.to_int


    # make the y_label to onehot
    def ylabel_to_onehot(self, Y):
        onehotY = np.zeros((Y.shape[0],2))
        for i in range(Y.shape[0]):
            onehotY[i][self.label2int(Y[i])] = 1

        return onehotY

#===========build model=====================

def bulid_model(X,Y,*list_index):

    l1,l2 = list_index
    batch_size = 32
    epochs = 30

    bank_input = Input(shape=(X.shape[1],))
    hidden = Dense(512)(bank_input)
    hidden = Dense(256)(hidden)
    hidden = Dense(128)(hidden)
    output = Dense(2, activation='softmax')(hidden)

    model = Model(inputs=bank_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    history = model.fit(X[l1], Y[l1],
             batch_size=batch_size,
             epochs=epochs,
             validation_data=(X[l2], Y[l2]),
             )

    return model


#===================train====================
def train_and_process(p_data):

    csv_df = p_data.read_csv(p_data.traindata)
    csv_df.iloc[:, :-1] = p_data.convert_to_value(csv_df.iloc[:, :-1])

    # make X values normalized
    csv_df.iloc[:, :-1] = p_data.normalized(csv_df.iloc[:, :-1])

    # spilt the data to train set and validation set
    # ,and get the list of split index
    X, Y, rs_list = p_data.cross_validation_split(csv_df)

    # get the first fold split
    l1,l2 = list(rs_list)[0]

    onehotY = p_data.ylabel_to_onehot(Y)
    model = bulid_model(X,onehotY,l1,l2)

    return model

#===================test========================

def test(p_data,model):

    test_df = p_data.read_csv(p_data.testdata)
    test_df = p_data.convert_to_value(test_df)
    test_df = p_data.normalized(test_df)

    y_prob = model.predict(test_df)
    y_classes = y_prob.argmax(axis=-1)
    sub_csv = pd.DataFrame({'id':np.arange(y_classes.shape[0]),
                            'ans':y_classes
                           },
                          columns=['id','ans'])

    result_name = 'sub_res.csv'
    sub_csv.to_csv(p_data.path+result_name, index=False)


#================main======================


def main():

    path = "/dataset/Bank Marketing Data Set/"
    train_data = "training_data.csv"
    test_data = "testing_data.csv"

    p_data = PreProcessing(path, train_data, test_data)

    model = train_and_process(p_data)
    test(p_data,model)


#========================================
if __name__=='__main__':
    main()