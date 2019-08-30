# DisasterResponse
Created a pipeline to filter and classify messages sent during disasters

## Table of Contents
1. [Description](#description)
2. [Run Your own Disaster Response Classifier!](#startup)
	1. [Packages needed](#packages)
	2. [Installing](#installing)
	3. [Executing Program](#executing)

<a name="descripton"></a>
## Description
This repository contains a ML pipeline that can classifier messages sent during disasters 

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

<a name="executing"></a>
### Executing Program:
1. The following commands will execute the program and create both pipelines that we need to run the web app. Make sure you are in the projects home directory when running

    - ETL Pipeline (cleans and stores data)
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - ML pipeline (trains classifier)
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command from the "app" directory to get your web app running.
    `python run.py`

3. Go to
