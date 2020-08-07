import utils
import joblib
import sys
import nltk

#This function loads the model data that is the model and the vocabulary from the model file path
#Input : model file path
#Returns the model and the vocabulary
def load_model(model_file_path):
	model_data = joblib.load(model_file_path)
	model = model_data['model']
	vocabulary = model_data['vocabulary']
	return model,vocabulary

#This function makes a prediction whether the test string contains a request or an issue or not
#Input: trained model file path and test string
#Returns the prediction the model makes on the test string (0 or 1) 
def predict(model_file_path,test_string):
	#load the model and vocabulary from the file
	model,vocabulary = load_model(model_file_path)
	#get a feature vector from the test string based on the vocabulary we obtained from the training data 
	test_data = utils.process_test_data(test_string,vocabulary)
	prediction = model.predict(test_data)
	return prediction[0]

#This functon finds the topic in the given test string if it contains a request or an issue
#Input : test string
#Returns the topic (string) in the given test string
def find_topic(test_string):
	#tag the words in the test string
	tagged_string = nltk.pos_tag(test_string[0])
	topic_string = ""
	#identifying the nouns in the test string
	for (word,tag) in tagged_string:
		if tag.find("NN") != -1:
			topic_string += " " +word
	return topic_string

if __name__ == "__main__":
	if len(sys.argv) != 2 :
		print(sys.argv[0], "takes 1 argument. Not ", len(sys.argv)-1)
		print("Arguments: \"string\" .The argument should be enclosed in double quotes ( \" ). Example:",sys.argv[0],'"Can I help?"')
		sys.exit()
	model_file_path = "final_model.sav"
	test_string = [sys.argv[1]]
	#predict whether the test string contains a request or an issue or not
	prediction = predict(model_file_path,test_string)
	if prediction == 0:
		print("Does the statement contain a request or an isssue? NO")
	else:
		print("Does the statement contain a request or an isssue? YES")
		#if the given test string contains a request or an issue, find the topic of the request or issue
		topic = find_topic(test_string)
		#if the topic is not identifiable print the entire test string as the topic
		if topic == "":
			print("here")
			print("Topic of the request or issue : "," ".join(test_string[0]))
		#else print the topic found	
		else:
			print("Topic of the request or issue : ",topic)


	