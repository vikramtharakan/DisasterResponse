# DisasterResponse
Created a pipeline to filter and classify messages sent during disasters

## Table of Contents
1. [Background](#background)
2. [Run Your own Disaster Response Classifier!](#startup)
	1. [Packages needed](#packages)
	2. [Installing](#installing)
	3. [Running Program](#run)

<a name="background"></a>
## Background
When a disaster strikes, there are a flood of messages that disaster response teams have to respond to. The goal of this project is to creat a classifier that is able to classifier these methods into different categories in order to help the response teams

In order to do this I first created an ETL pipeline that extracts the data, cleans it, and then uploads it to a database. Then I created a machine learning pipeline that loads the data from the database, and then trains a multioutputclassifier model on the text from the database. From here I was able to manipulate parameters to get my final accuracy score up to about 93%. 

<a name="startup"></a>
## Run Your own Disaster Response Classifier!

<a name="packages"></a>
### Packages needed
- Python (3.5)
- pandas
- numpy
- nltk
- sciki-Learn
- SQLalchemy
- Flask
- Plotly

### Installing
Simply download these files directly or clone this repository using git. 

<a name="run"></a>
### Running Program:
1. The following commands will execute the program and create both pipelines that we need to run the web app. Make sure you are in the projects home directory when running

    - ETL Pipeline (cleans and stores data)
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - ML pipeline (trains classifier)
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command from the "app" directory to get your web app running.
    `python run.py`

3. Go to http://0.0.0.0:3001/ to see your running web app! From here you'll be able to type in disaster messages of your own and see how they get classified.
