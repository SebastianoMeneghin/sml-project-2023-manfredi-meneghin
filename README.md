# Scalable Machine Learning and Deep Learning, Final Project, 2023/2024

## About
This repository is related to the **Final Project** of the course [ID2223 Scalable Machine Learning and Deep Learning](https://www.kth.se/student/kurser/kurs/ID2223?l=en) at [KTH](https://www.kth.se). The **project proposal** can be found [here](https://github.com/SebastianoMeneghin/sml-project-2023-manfredi-meneghin/blob/main/SML_FinalProject_ProjectProposal_ManfrediMeneghin.pdf).

*"DeLight" - Delayed fLights*, our final project, consist in a *Serverless Machine Learning pipeline* able to *predict the flight delay* of flight daily departing from [Stockholm Arlanda International Airport](https://www.swedavia.se/arlanda/), depending on the weather condition and historical flight delay information.
In **this repository** you can find:
- [**Machine Learning Serverless Pipeline**](https://github.com/SebastianoMeneghin/sml-project-2023-manfredi-meneghin/tree/main/src) comprehensive of running daily script for both [local](https://github.com/SebastianoMeneghin/sml-project-2023-manfredi-meneghin/tree/main/src/local_daily_scripts) and remote environments.
- [**Flight Info and Meteorological Analysis (MESAN) in Arlanda**](https://github.com/SebastianoMeneghin/sml-project-2023-manfredi-meneghin/tree/main/datasets/) of the full year 2023.
- [**Graphical User Interface on Gradio**](https://github.com/SebastianoMeneghin/sml-project-2023-manfredi-meneghin/tree/main/src/hugging_face_user_interface) running on [*HuggingFace*](https://huggingface.co/spaces/SebastianoMeneghin/flight_delay) 🤗, where you can play with the data (our own version of [**tl:dr**](https://huggingface.co/spaces/SebastianoMeneghin/flight_delay))

The project has been developed in **Python** on our own machines, but we have used **Jupyter notebooks** for Exploratory Data Analysis for a more insightful data visualization.


### The Team

* [Giovanni Manfredi](https://github.com/Silemo)
* [Sebastiano Meneghin](https://github.com/SebastianoMeneghin)


## Table of Contents
* [Introduction](#Introduction)
* [Architecture](#Architecture)
* [Pipelines description](#Pipelines-description)
* [Results](#Results)
* [How to run](#How-to-run)
* [Software used](#Software-used)


## Introduction
The project can be divided into **four main pipelines**:
- **Historical data collection and preparation (historical feature pipeline)**: the features on weather condition and flight information have been extracted from [different source](#Results), such as *API Vendors* and *institutional OpenData archives*. Then, data have been processed, studied and uniformed to be used *for training*. The data are saved remotely in [Hopsworks Feature Store](https://www.hopsworks.ai/).
- **Daily data collection (real-time feature pipeline)**: everyday new data are extracted from various sources and processed, accordingly to the *standards and rules* set by the historical data, to be used for data augmentation and quotidian model training.
- **Model training and evaluation (training pipeline)**: a new model is trained everyday, with a bigger amount of data, thanks to the new data gathered in the second pipeline and to daily scripts running remotely on [Modal](https://modal.com/). The model is then saved in Hopsworks Model Registry.
- **Batch prediction and result visualisation (inference pipeline)**: everyday at midnight the daily forecast and flight schedules for the two following days are collected by a remote script. The model is accessed from the remote storage and delay predictions are made. The prediction results can be accessed through a [*user interface*](https://huggingface.co/spaces/SebastianoMeneghin/flight_delay) hosted on [*HuggingFace*](https://huggingface.co/) 🤗 created with [*Gradio*](https://www.gradio.app/). This application allows to use and see the model's functionalities.


## Architecture
<img alt="Machine Learning Pipeline Schema" src="/images/pipeline.png" >


## Pipelines description
### Historical feature pipeline
[**Here**](https://github.com/SebastianoMeneghin/sml-project-2023-manfredi-meneghin/tree/main/src/feature_pipeline/feature_pipeline_historical) is where the **historical data** are collected, in order to have a feature base to train our machine learning algorithm. The **typology and sources** of our data are:
- **Meteorological Analysis** collected through [SMHI OpenData Grid Archive](https://opendata-download-grid-archive.smhi.se/feed/6)
- **Flight Information** collected through [Zyla API Hub - Historical Flights Information API](https://zylalabs.com/api-marketplace/data/historical+flights+information+api/1020/)

Due to the need of collecting a **big amount of heterogenous data**, saved in different formats, such as ```GRIB``` for Meteorological Data and ```.json``` for Flight Information, the feature pipeline is divided in *weather*, *flight* and *merged* feature pipelines.

Both the **[*weather*](https://github.com/SebastianoMeneghin/sml-project-2023-manfredi-meneghin/tree/main/src/feature_pipeline/feature_pipeline_historical/feature_pipeline_historical_weather) and [*flight*](https://github.com/SebastianoMeneghin/sml-project-2023-manfredi-meneghin/tree/main/src/feature_pipeline/feature_pipeline_historical/feature_pipeline_historical_flight) feature pipeline** are divided in the following:
- **Collector:** *data from every day and every hour are iteratively collected* from respectively the SMHI online library and through a Flight Info API (Zyla Flight API can be replaced by competitors alternatives). The *data are locally saved* before to be dealt with.
- **Extractor:** *data from each saved file are extracted* from their own format, using the Python library [```pygrib```](https://pypi.org/project/pygrib/) to access meteorological ```GRIB``` files. All the raw data for both weather and flight data are *saved into different files ```.csv```*.
- **Processor:** *data are transformed into a shared uniform format* in order to be studied and be used for training. For instance, *datetimes are split* of in year, month, day and hour, depending on the need; wind *direction and coordinates are rotated* to standard configuration. Finally, *timezone is set to Stockholm*, with DST, in both dataset.

Then the **two dataset created are merged**, through the script [*Dataset Merger*](https://github.com/SebastianoMeneghin/sml-project-2023-manfredi-meneghin/blob/main/src/feature_pipeline/feature_pipeline_historical/feature_pipeline_flightWeather_historical_merger.py).  Once the dataset has been created, we pass through an [*Exploratory Data Analysis*](https://github.com/SebastianoMeneghin/sml-project-2023-manfredi-meneghin/blob/main/src/feature_pipeline/feature_pipeline_historical/feature_pipeline_flightWeather_EDA.py), in order to evaluate which factor influence most the flight delay. More on that in the [Results](#Results).

Selected the right feature with the best attributes (*and some other promising for a future when will have more data*) are **uploaded** with the last script of the historical pipeline chain, called [*Dataset Uploader*](https://github.com/SebastianoMeneghin/sml-project-2023-manfredi-meneghin/blob/main/src/feature_pipeline/feature_pipeline_historical/feature_pipeline_flightWeather_historical_uploader.py). Through this procedure, the file are uploaded and saved in a dedicated [*Hopsworks' Feature Group*](https://docs.hopsworks.ai/feature-store-api/2.5.9/generated/feature_group/).

### Realtime feature pipeline
**Everynight a scripts runs remotely** on [Modal](https://modal.com/), acquiring the daily schedule and the meteorological analysis of two days before. [*Backfill Feature Pipeline script*](https://github.com/SebastianoMeneghin/sml-project-2023-manfredi-meneghin/blob/main/src/feature_pipeline/feature_pipeline_realtime/backfill_feature_pipeline_flightWeather_daily.py) **cleans and transforms the data** into the project format and saves them on the Feature Group. Thanks to that, our dataset is constantly evolving, making it for real a **dynamic source of data**.

The **meteorological analysis** are extracted as the historical data through [SMHI OpenData Grid Archive](https://opendata-download-grid-archive.smhi.se/feed/6), but also through the [SMHI OpenData Meteorological Analysis API](https://opendata.smhi.se/apidocs/metanalys/index.html). Instead, the [Swedavia Flight Info v2 API](https://apideveloper.swedavia.se/api-details#api=flightinfov2&operation=5bf658bdbc86470e887fb301) is used to *access* **flight information** of departed flight from [Stockholm Arlanda International Airport](https://www.swedavia.se/arlanda/).

### Training pipeline
Greatest part of the feature engineering, including several *model-independent transformation* said before, has been done. However, having selected a [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) framework called [```XGBoost```](https://xgboost.readthedocs.io/en/stable/index.html) as our own **regression model**, we still need to do some *model-dependent transformation*, as *binning and labelling* some attributes, *remove attribute-specific outliers* or create some *dummy variables*.

**Set up the data standard**, we can start the [*Model Evaluation and Selection*](https://github.com/SebastianoMeneghin/sml-project-2023-manfredi-meneghin/blob/main/src/training_pipeline/training_pipeline_flightWeather_model_selection_and_evaluation.py) process, where thanks to [```scikit-learn```](https://scikit-learn.org/stable/index.html) module called ```GridSearchCV``` we *tune hyperparameters* for our own model:

| Name        | Best Value   |
| :---------: | :-----:      |
|n_estimators | 45           |
| max_depth   | 15           |
| eta         | 0.05         |
| subsample   | 0.85         |

Then a first version of the model is trained on the data accessed on the [Hopsworks' Feature Group](https://docs.hopsworks.ai/feature-store-api/2.5.9/generated/feature_group/), and uploaded in the [Hopsworks' Model Registry](https://docs.hopsworks.ai/3.5/concepts/mlops/registry/), through the [*Initializer*](https://github.com/SebastianoMeneghin/sml-project-2023-manfredi-meneghin/blob/main/src/training_pipeline/training_pipeline_flightWeather_initializer.py). **Everynight a scripts runs remotely** on [Modal](https://modal.com/), getting the data from the feature store the old data and the new acquired data from the day. The [*Daily Training Pipeline script*](https://github.com/SebastianoMeneghin/sml-project-2023-manfredi-meneghin/blob/main/src/training_pipeline/training_pipeline_flightWeather_daily.py) trains **a new model** and save it in the model registry, **replacing the previous version**.

### Inference pipeline
**Everynight at midnight** a scripts runs remotely on [Modal](https://modal.com/), and **predicts the departure's delay of flight** departing from [Stockholm Arlanda International Airport](https://www.swedavia.se/arlanda/), for the day itself and the following one. 
This is done by collecting through [Swedavia Flight Info v2 API](https://apideveloper.swedavia.se/api-details#api=flightinfov2&operation=5bf658bdbc86470e887fb301) the **flight information**, while the **meteological forecast** are accessed through the last API of this long list, [SMHI Open Data Meteorological Forecast API](https://opendata.smhi.se/apidocs/metanalys/index.html).

Collected the data, the last trained version of the **regression model** is downloaded from our [Hopsworks' Model Registry](https://docs.hopsworks.ai/3.5/concepts/mlops/registry/) and the **new predictions are calculated**. Those **are saved into** [Hopsworks' File System](https://www.hopsworks.ai/), replacing the day before prediction.


## Results
### Graphic User Interface
The **results of our work can also be 👀 seen** out of [*this repo*](https://en.wikipedia.org/wiki/Recursion), by 🛬 landing on a fun [*🛩️ User Interface✈️*](https://huggingface.co/spaces/SebastianoMeneghin/flight_delay) hosted on [HuggingFace](https://huggingface.co/)🤗. With three different tabs, you can decide to:

🛩️ Select a [specific flight](https://huggingface.co/spaces/SebastianoMeneghin/flight_delay), by **answering to some absolutely relevant questions**.
✈️ View the [full schedule](https://huggingface.co/spaces/SebastianoMeneghin/flight_delay) of today or tomorrow flights, with respective delay.
📊 Took a glance of [model performances and dataset size](https://huggingface.co/spaces/SebastianoMeneghin/flight_delay), with daily updates.

All the files needed to **recreate the same graphical interface** or test your own with a synthetic dataset can be found in the [*User Interface folder*](https://github.com/SebastianoMeneghin/sml-project-2023-manfredi-meneghin/tree/main/src).

### Model Performance
Since the fisrt EDA and then through further analysis the **data showed a low correlation** (<20% for whichever variable) **between flight schedules, wheather conditions and flight delay**. However, from the starting point, **significant improvements have been obtained** by *feature selection*, *binning variables* and creating some *dummy variables*, especially with the most incisive variables (e.g. wind from south-east, temperature below -20°C, local flights). Analyzing the data we have discovered that such phenomena just described are **more an expection than a recurrent event**, so we can assume that those few feature are not numerous enough to influence the model significantly.

Another significant factor is the **arguably low number** of features collected, since the data consists only in 2023 departing flight from [Stockholm Arlanda International Airport](https://www.swedavia.se/arlanda/).

**Future improvement could be pursued** by waiting for a *bigger amount of feature*, as well as *more intense model tuning*, *model substitution* with different models or *adding new features* collecting data from **other sources with more various data** (e.g. number of arrivals around departing time)


## How to run
The  whole project has been developed with **high-attention for replicability** and future-proofing. Indeed, **it is possible to replicate this whole project** by only running scripts **locally on your own laptop** with Python installed on, without the use of any other platform, as [Modal](https://modal.com/) or [Hopsworks](https://www.hopsworks.ai/).

In order to be ready, you first need to *set up your own* **environment**, according to the file ```environment.yml```. This can be done easy via ```conda```, through the command ```conda env create --file environment.yml``` 

Then, you will need to **grant access to a flight information and a weather forecast/analysis provider**, so you will need to get the API Key from that service. [Swedavia APIs](https://apideveloper.swedavia.se/) and [SMHI OpenData APIs](https://opendata.smhi.se/apidocs/) are our **suggested free option**, for flights from and to Sweden and weather information in a broader area above the Scandinavian countries. About the **historical data**, you can access easily to all the stages of the data cleaning and processing in [Datasets](https://github.com/SebastianoMeneghin/sml-project-2023-manfredi-meneghin/tree/main).

*Roll up your sleeves. Now you are ready for real!*






### Software used
[**Visual Studio Code**](https://code.visualstudio.com/) - main IDE

[**GitKraken**](https://www.gitkraken.com/) - git versioning

[**Google Colab**](https://colab.research.google.com/) - running environment

[**HuggingFace**](https://huggingface.co/) - dataset, model registry, GUI

[**Gradio**](https://www.gradio.app/) - GUI

[**Modal**](https://modal.com/) - run daily remote script

[**Hopsworks**](https://www.hopsworks.ai/) - MLOps platform

[**Zyla API**](https://zylalabs.com/) - historical flight API

[**Swedavia API**](https://apideveloper.swedavia.se/) - real-time flight API

[**SMHI API**](https://opendata.smhi.se/apidocs/metfcst/index.html) - real-time forecast API

[**SMHI OpenData**](https://opendata-download-metanalys.smhi.se/) - historical meteorological API
