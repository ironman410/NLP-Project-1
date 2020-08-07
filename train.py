from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import utils
import joblib

#This function initializes and trains the random forest classifier
#Input: the feature matrix, the labels and the number of estimators for the random forest classifier
#Returns the trained model
def train_classifier(train_x,train_y,n_estimators=100):
	#initialising the Random Forest Classifier
	model = RandomForestClassifier(n_estimators=n_estimators)
	#Training the random forest classifier with the training data
	model.fit(train_x,train_y)
	return model

#This function stores the model and vocabulary in an sav file using joblib
#Input : trained model and vocabulary
#Returns nothing
def store_model(model,vocabulary):
	#the model file name is predefined
	model_file = 'final_model.sav'
	#store the model as well as the vocabulary which is required for prediction
	joblib.dump({'model': model,'vocabulary':vocabulary},model_file)

#This funstion trains the classifier and stores the trained model
#Input: training file path
#Returns nothing
def train_and_store(train_file_path):
	#process training data to get feature matrix and labels
	train_x,train_y,vocabulary_dictionary = utils.process_training_data(train_file_path)
	#train the model based on the processed training data
	trained_model = train_classifier(train_x,train_y)
	#store the model and the vocabulary
	store_model(trained_model,vocabulary_dictionary)
	y_pred = trained_model.predict(train_x)
	print("Training Accuracy: ",metrics.accuracy_score(train_y,y_pred))

if __name__ == "__main__":
	train_file_path = "train.csv"
	#train the classifier and store the trained model
	train_and_store(train_file_path)
	