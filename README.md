# NLP-Project-1
# Problem Statement

This Project deals with the problem statement stated below :

Given a text message, the code that you write should be able to determine 
1. if the message contains a request for help or having an issue that needs to be resolved 
2. the topic on which the request / issue is seeking resolution.

# Instructions for Running the Project

In order to run this project download the files :
1. final_model.sav
2. test.py
3. utils.py

*Note : Make sure that all the files are in the same folder when running the program.*

The program can be run from the command line with following command :

python test.py "text message>"

*Note:* 

*1. Make sure that the text message is enclosed in double quotes.*

*2. The program runs without any errors on Python version 3.7.6.*

# Libraries used
The libraries used for this project include :
1. NLTK
2. JobLib
3. Numpy
4. Scikit-Learn

# Assumptions
The following assumptions were made while constructing this project :

1. Customers communicate in English only.

2. Customers use gramatically correct sentences.

3. Customers use only certain vocabulary to indicate an issue or a request.

4. Customers spell correctly (atleast in most cases).

5. The nouns in the text message are indicative of the topic of the issue or request.

6. There is only a single request or issue specified in a given text message.

# Dataset

The dataset "ticket-master.csv" was obtained from the internet. The dataset contained 5000 rows and was unlabelled. A subset of this dataset containing only 2000 rows was used as the training dataset and was labelled manually. Only the title column of the original dataset was used in training data. The label 1 was used to indicate that the record has an issue or a request and the label 0 was used to indicate that neither of them were present. Text messages in other language other than English were labelled as 0 based on the assumption that all communication the program was dealing with is in English.

# Processing the Training data

The training dat was cleaned that is punctuations, numbers and extra white space characters were remove. Further, the dataset was tokenized and the vocabulary consisting of unigrams and bigrams that were not rare were extracted. The feature vector built where each column in each row indicated the presence of a particular unigram and bigram in the training record.

# Approach for First Part of the Problem

The first part of the problem required us to find whether a particular text message contained a request or an isssue or not. Supervised Learning was used to solve this problem. The records were manually labelled and sent through a supervised learning algorithm. The Learning algorithm used was Random Forest. Due to the lack of amount of data available a neural network was not used to solve this problem.

# Approach for Second Part of the Problem

The second part of the problem required us to find the topic of the request or issue. To solve this problem, part of speech tagging was done on the message and the nouns were extracted from the same. This was done because afer observing the dataset and reading articles on information extraction, it is found thatif a sentence is constructed properly the  noun mostly indicates what is being talked about in the sentence. So, in this case the noun present in the message may mostly be the topic of the issue or the request.

# Results
The accuracy on the training set was **83.95 %**. Some of the screenshots of the output are shown below :

# Other Probable Approaches to the Problem

One other solution to the first part of the problem could be to use unsupervised learning methods to learn from the data instead of labelling it manually. Due to the relatively small amount if data available, manually labelling the data was preferred. For the second part of the problem, apart from the nouns the verbs could also be extracted to give a better understanding of the topic of the problem. Since the data available had small sentences, and in most cases extracting both the nouns and verbs would result in just displaying the message again , this approach of extracting nouns was preferred.

# Future Work 

This project can be further extended by doing the following :

1. Collecting more data from customer service interactions and using it to train the algortihm.

2. Identifying the language of communication instead of assuming that the language would always be English and translating the message to English so that the same rules can be used for a different language.

3. Adding more features like trigrams and skip grams so that the learning algorithm is more robust.

4. Adding word features like presence of capital words to not only find out the topic but also find the sentiment of the customer to indicate the priority of his request or issue.

5. Adding features like the presence of question words, presence of question marks to indicate the presence of a request or an issue in the message.
