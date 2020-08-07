import csv
import json
import re
import string
import numpy as np
import nltk
import collections
from nltk.tokenize import word_tokenize

#This function loads the train file given the train file path
#Input: train file path
#Returns train_x and train_y which are lists of records and labels in the given dataset 
def load_train_file(train_file_path):
	train_file = open(train_file_path,'r',encoding="utf-8")
	reader = csv.reader(train_file, delimiter=',')
	train_x = []
	train_y = []
	header = next(reader)
	for row in reader:
		train_x.append(row[1])
		train_y.append(int(row[2]))
	return train_x,train_y

#This function cleans the text specifically converts all letters to lower case, removes numbers, removes punctuations and removes white spaces
#Input : list of training records (train_x)
#Returns list of cleaned training records
def clean_text(train_x) :
	for index,x in enumerate(train_x):
		#convert text to lowercase
		train_x[index] = train_x[index].lower()
		#removing numbers 
		train_x[index] = re.sub(r'\d+','',train_x[index])
		#removing symbols
		train_x[index] = train_x[index].translate(str.maketrans(string.punctuation," "*len(string.punctuation)))
		#removing white spaces
		train_x[index] = train_x[index].strip()
	return train_x

#This function tokenizes the records that is it splits the records into tokens ( words )
#Input : list of training records (train_x)
#Returns list of tokenized records
def tokenize(train_x):
	for index,x in enumerate(train_x):
		tokens = word_tokenize(x)
		train_x[index] = tokens
	return train_x

#This function builds the vocabulary from the tokenized dataset and removes the words that have a frequency less than the given threshold.
#Input : list of tokenized records and the frequency threshold
#Returns the vocabulary dictionary where the key is the word and the value is the index for the particular word
def get_vocabulary(train_x,threshold=1):
	vocabulary_freq_dict = {}
	#finding the frequency of words in the dataset
	for x in train_x:
		for i,word in enumerate(x):
			#finding unigrams
			if word not in vocabulary_freq_dict:
				vocabulary_freq_dict[word] = 1
			else :
				vocabulary_freq_dict[word] += 1
			#finding bigrams
			if i <= len(x)-2:
				if (x[i],x[i+1]) not in vocabulary_freq_dict:
					vocabulary_freq_dict[(x[i],x[i+1])] = 1
				else :
					vocabulary_freq_dict[(x[i],x[i+1])] += 1
	#Removing rare words from the vocabulary that is removing words that occur less than the given threshold
	vocabulary_dictionary = {}
	current_index = 0
	for word,frequency in vocabulary_freq_dict.items() :
		if frequency > threshold :
			vocabulary_dictionary[word] = current_index
			current_index +=1
	return vocabulary_dictionary

#This function takes the tokenized sentences and converts into a feature vectors depending upon the vocabulary
#Input: list of tokenized sentences and the vocabulary dictionary
#Returns the feature matrix 
def get_feature_matrix(train_x,vocabulary_dictionary):
	#creating an empty feature matrix where the number of rows is equal to the number of training records and the number of columns is the number of words in the vocabulary
	feature_matrix = np.empty([len(train_x),len(vocabulary_dictionary)])
	for index,x in enumerate(train_x):
		feature_vector = np.zeros(len(vocabulary_dictionary))
		for i,word in enumerate(x):
			word_index = vocabulary_dictionary.get(word)
			if(word_index != None) :
				feature_vector[word_index] = 1
			if i <= len(x)-2:
				bigram_index = vocabulary_dictionary.get((x[i],x[i+1])) 
				if(word_index != None) :
					feature_vector[bigram_index] = 1
		feature_matrix[index] = feature_vector
	return feature_matrix			

#This function processes the entire training data
#Input : training file path (csv file)
#Returns the feature matrix ,the labels and the vocabulary ( which needs to be used for testing)
def process_training_data(train_file_path):
	#loading the training file
	train_x,train_y =load_train_file(train_file_path)
	#cleaning the text
	train_x = clean_text(train_x)
	#tokenizing the text
	train_x = tokenize(train_x)
	#getting the vocabulary from the dataset that does not contain rare words
	vocabulary_dictionary = get_vocabulary(train_x)
	#converting the tokenized sentences into feature vectors based on the vocabulary previously obtained
	train_x = get_feature_matrix(train_x,vocabulary_dictionary)
	return train_x,train_y,vocabulary_dictionary

#This function processes the test data where the data is a string 
#Input : test string and the vocabulary
#Returns the feature vector
def process_test_data(test_string,vocabulary):
	#cleaning the text
	cleaned_test = clean_text(test_string)
	#tokenizing the text
	tokenized_test = tokenize(cleaned_test)
	#converting the tokenized sentences into feature vectors based on the vocabulary 
	test_x = get_feature_matrix(tokenized_test,vocabulary)
	return test_x

