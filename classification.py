#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import import_ipynb
import testsets
import evaluation

# TODO: load training data
import re
import nltk
import numpy as np
import csv
import pandas as pd
from nltk.stem import WordNetLemmatizer
train_data = []
tweet_list = []
wordsdictionary = []
word_lemma = WordNetLemmatizer()

def readfile(filepath):
    
    column_names = ["id", "label", "tweet"]
    filename = pd.read_csv(filepath, delimiter = '\t', names=column_names, header=None)
    return filename
    
def preprocessing(dataframename):
    dataframename['tweet']  =  dataframename['tweet'].str.replace("#[\w]*","")#replaces anyword or string preceded by # tag with empty string
    dataframename['tweet']  =  dataframename['tweet'].str.replace("@[\w]*","")# replaces @usernames with empty string
    dataframename['tweet']  =  dataframename['tweet'].str.replace("(http(s)?://|www\.|ftp://)+([a-zA-Z0-9]+[^a-zA-Z0-9]*)*","")#replaces URL's with empty string          
    dataframename['tweet']  =  dataframename['tweet'].str.replace("\<\3","love") 
    dataframename['tweet']  =  dataframename['tweet'].str.replace("\:\)","happy")
    dataframename['tweet']  =  dataframename['tweet'].str.replace("\:\D","happy")
    dataframename['tweet']  =  dataframename['tweet'].str.replace("\:\(","sad")
    dataframename['tweet']  =  dataframename['tweet'].str.replace("\:\(\(","sad")
    dataframename['tweet']  =  dataframename['tweet'].str.replace("\:\'\(","sad") 
    dataframename['tweet']  =  dataframename['tweet'].str.replace("🤗","Happy hugging face")
    dataframename['tweet']  =  dataframename['tweet'].str.replace("👍","Victory")
    dataframename['tweet']  =  dataframename['tweet'].str.replace("👏","Happy") 
    dataframename['tweet']  =  dataframename['tweet'].str.replace("😍","Happily beaming face")      
    dataframename['tweet']  =  dataframename['tweet'].str.replace("😂","Tears of joy")
    dataframename['tweet']  =  dataframename['tweet'].str.replace("([^\s\w]|_)+","")#replaces alphanumeric with empty string        
    dataframename['tweet']  =  dataframename['tweet'].str.replace("r' +'","")#removes the empty spaces 
    dataframename['tweet']  =  dataframename['tweet'].str.replace("(^|\s)[0-9]+(\s|$)","")# replaces number with empty string
    dataframename['tweet']  =  dataframename['tweet'].str.replace(r"\b[a-zA-Z]{1}\b","")#replaces single characters with empty string
    dataframename['tweet']  =  dataframename['tweet'].apply(lambda x: ' '.join([i for i in x.split() if len(i)>3]))# removing short words
    cleaned_tweet = dataframename['tweet'].apply(lambda x: x.lower().split())#tokenization of tweet
    cleaned_tweet = cleaned_tweet.apply(lambda x: [word_lemma.lemmatize(i) for i in x]) # stemming
    for elem in range(len(cleaned_tweet)):
        cleaned_tweet[elem] = ' '.join(cleaned_tweet[elem])
    dataframename['tweet']  = cleaned_tweet
    return dataframename

######### Reading training set file and Feature extraction using bag-of-words########

traindf = readfile("Desktop/semeval-tweets/"+"twitter-training-data.txt") #reading the train data file
traindf_preprocessing = preprocessing(traindf)#preprocessing the training set
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
#The approach to Convert words into numbers is called Bag-of-words, 
#Bag of words ignores grammer, create a vocabulary for entire corpus/document
#and then based on the created vocabulary, each document or file in corpus converts into vectors
vectorizer = CountVectorizer(stopwords.words('english'),ngram_range =(1,2))
#count vectorization is an bag of words approach, counts the frequency of the words ocuured in the entire corpus/document
#tf - idf (term frequency - Inverse Document frequency ) an approach to find the important sentences in a document/corpus.
#tf - idf provides the words that are frequently occured and holds high weightage in the document.
vectorizer.fit(traindf_preprocessing['tweet'])
vector = vectorizer.transform(traindf_preprocessing['tweet'])
tf_transformer = TfidfTransformer().fit(vector)
train_tfidf = tf_transformer.transform(vector)
y_train = traindf.iloc[:,1].values

################ Tokenization and pad_sequence on traiing set using keras#########

from keras.models import Sequential,load_model
from keras.layers import Dense,LSTM,Embedding,Dropout,Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# using tokenizer from keras module , we can convert the twitter data into sequences so that we can pass that into embedding matrix

tokenizer = Tokenizer(num_words = 5000)#Tokenizer in keras helps to convert the text into sequence of numbers based on tf-idf or wordcount.
tokenizer.fit_on_texts(traindf_preprocessing['tweet'])
sequences = tokenizer.texts_to_sequences(traindf_preprocessing['tweet'])# converting into sequence of vectors
data_seq = pad_sequences(sequences, maxlen = 300) # pad_sequences ensures that all the sequences has the same length ,when max length wasnot mentioned the pad_sequence takes the lenth of longest sequence

############# Reading a glove file and creating embedding matrix##############

#word embedding is an approach where we represent the words in dense vectors representation, where vectors represents the position of word. 
#we use word embedding because tf-idf gives sparse vectors where it represents the each word with in a vector of entire vocabulary.
embeddings_dict = {}
with open("Desktop/glove.6B/glove.6B.100d.txt", 'r') as embeddingsfile:#opening glove file
    for line in embeddingsfile:#reading the each line in glove 100 dimensional file
        values = line.split()#splitting the values in each file
        
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        # storing the words and vectors  in a dictionary , words as keys and vectors as values
        embeddings_dict[word] = vector
words = min(5000, len(tokenizer.word_index) + 1) 
embedding_matrix = np.zeros((words,100))# created a embedding matrix with zeros 
for word, index in tokenizer.word_index.items():#iterating over dictionary
    if index >= 5000:# if index id greater than or equal to 5000 dont append into embedding_matrix
            break
    else:
        embedding_vector = embeddings_dict.get(word)#if the index is with in 5000 then get the words from embedding_dict and append into embedding matrix
        if embedding_vector is not None:
             embedding_matrix[index] = embedding_vector
            
###### Building Predictive models##########
   

for classifier in ['Multinomial NaiveBayes', 'Linear SVC', 'LSTM']: # You may rename the names of the classifiers to something more descriptive
    if classifier == 'Multinomial NaiveBayes':
        print('Training ' + classifier)
        # TODO: extract features for training classifier1
        # TODO: train sentiment classifier1
        from sklearn.naive_bayes import MultinomialNB
        classifier_model = MultinomialNB(alpha = 1.0,fit_prior = False)
        classifier_model.fit(train_tfidf,y_train)
        
    elif classifier == 'Linear SVC':
        print('Training ' + classifier)
        # TODO: extract features for training classifier2
        # TODO: train sentiment classifier2
        from sklearn.svm import LinearSVC
        classifier_model = LinearSVC(C=1.0,tol = 1e-4,multi_class='crammer_singer')
        classifier_model.fit(train_tfidf,y_train)
        
    elif classifier == 'LSTM':
        print('Training ' + classifier)
        # TODO: extract features for training classifier3
        # TODO: train sentiment classifier3
        from sklearn.preprocessing import LabelBinarizer
        binarizer = LabelBinarizer()
        binarizer.fit(traindf_preprocessing['label'])
# LabelBinarizer helps in converting coverting each label such as positive, negative ,neutral in twitter sets to unique numbers 
         
        y = binarizer.transform(traindf_preprocessing['label'])
        model = Sequential()
        model.add(Embedding(5000,100,weights=[embedding_matrix],input_length = 300,trainable=False))
        model.add(LSTM(100,dropout = 0.2))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        #one epoch is very huge to go forward and backwards in neural network 
        #so we didvide the data into multiple sets and epoch helps in increasing the model accuracy.
        model.fit(data_seq,y,epochs =3,verbose = 1)
        
################ Reading tessets and predicting the model##################     
    
    
    for testset in testsets.testsets:
        #TODO: classify tweets in test set
        testset = "Desktop/semeval-tweets/"+testset# passing test file path as parameter to readfile function
        test_dataset = readfile(testset)
        test_dataset = preprocessing(test_dataset)
                
        if classifier == 'LSTM':
            
            y_test = binarizer.transform(test_dataset['label'])
            sequences = tokenizer.texts_to_sequences(test_dataset['tweet'])# converting into sequence of vectors
            data_seq_test = pad_sequences(sequences, maxlen = 300) 
            #predicting the results on test set
            y_pred_model = model.predict(np.array(data_seq_test))
            #Inverse transform functions helps in converting the sparse matrix to categorical labels.
            pred_label = binarizer.inverse_transform(y_pred_model)
            predictions = {} # predictions dictionary for classifier 3
            for i in range(len(data_seq_test)):
            
                predictions[str(test_dataset['id'][i])] = str(pred_label[i])
            
        else:
            
            y_test = test_dataset.iloc[:, 1].values
            def testdata_vectorizing(test_daframename):
                df_count = vectorizer.transform(test_daframename['tweet'])
                df_tfidf = tf_transformer.transform(df_count)
                return df_tfidf
            
            test_tfidf = testdata_vectorizing(test_dataset)
            y_pred = classifier_model.predict(test_tfidf)
            
            predictions = {}#predictions dictionary for classifier 1 and 2
            for i in range(len(y_pred)):
                
                predictions[str(test_dataset['id'][i])] = y_pred[i]
        
             
                
        evaluation.evaluate(predictions,testset,classifier)
   
        evaluation.confusion(predictions,testset,classifier)


# In[ ]:




