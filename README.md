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
