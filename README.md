# Disaster Response Pipeline Project
## Project Overview
In this project I analyzed the disaster data that I received from Figure Eight and built a ETL and ML pipeline to classify the messages that were sent during disaster events.  The ML pipeline is a model for an API that classifies the messages based on their content. The dataset consists of the real messages that has been stored in Figure Eight database. By reorganizing and cleaning the dataset through ETL process along with saving and importing processed data as SQLite database, I started working on ML pipeline step. After further analyzing the dataset and dropping unrelated columns, I tried out several ML models, within ML pipeline including DecisionTreeClassifier, MLPClassifier, and ExtraTreesClassifier and picked the one with best performance and saved the trained model as pickle file.  The ML pipeline that I built categorize these messages so that they can be sent to an appropriate disaster relief agency. It is worth noting that the message dataset has 36 pre-defined categories and ML model will classify the new messages based on its content and categorize the new incoming messages under one of those predefined categories. The model was finally, used to build a web app where an emergency worker can input a new message and get classification results in several categories.
Below are a few screenshots of the web app.


 
 
## Installation
The project must be run with Python 3. The list of libraries for this project are as follow: pandas, numpy, sqlite3, nltk, re, sqlalchemy, sklearn, pickle, flask and plotly. 
## File Descriptions
The file structure of the project is given below: 
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
## Instructions
1. Run the following commands in the project's root directory to set up your database and model:
    - To run ETL pipeline that cleans data and stores in database
        `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app.
    `python run.py`
3. Go to http://0.0.0.0:3001/
## Acknowledgements:
Special thanks for both Udacity and Figure Eight team that made performing this task possible. 

