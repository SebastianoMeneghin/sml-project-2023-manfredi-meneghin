# Scalable Machine Learning and Deep Learning, Final Project, 2023/2024

## About
This repository is related to the **Final Project** of the course [ID2223 Scalable Machine Learning and Deep Learning](https://www.kth.se/student/kurser/kurs/ID2223?l=en) at [KTH](https://www.kth.se). The **project proposal** can be found [here](https://github.com/SebastianoMeneghin/sml-project-2023-manfredi-meneghin/blob/main/SML_FinalProject_ProjectProposal_ManfrediMeneghin.pdf).

*JetInTime*, our final project, consist in a *Serverless Machine Learning pipeline* able to *predict the flight delay* of flight daily departing from [Stockholm Arlanda International Airport](https://www.swedavia.se/arlanda/), depending on the weather condition and historical flight delay information.
In **this repository** you can find:
- [**Machine Learning Serverless Pipeline**](https://github.com/SebastianoMeneghin/sml-project-2023-manfredi-meneghin/tree/main/src) comprehensive of running daily script for both [local](https://github.com/SebastianoMeneghin/sml-project-2023-manfredi-meneghin/tree/main/src/local_daily_scripts) and remote environments.
- [**Flight Info and Meteorological Analysis (MESAN) in Arlanda**](https://github.com/SebastianoMeneghin/sml-project-2023-manfredi-meneghin/tree/main/datasets/) of the full year 2023.
- [**Graphical User Interface on Gradio**](https://github.com/SebastianoMeneghin/sml-project-2023-manfredi-meneghin/tree/main/src/hugging_face_user_interface) running on [*HuggingFace*](https://huggingface.co/spaces/SebastianoMeneghin/flight_delay) 🤗, where you can play with the data.

The project has been developed in **Python** on our own machines, but we have used **Jupyter notebooks** for Exploratory Data Analysis for a more insightful data visualization.

### The Team

* [Giovanni Manfredi](https://github.com/Silemo)
* [Sebastiano Meneghin](https://github.com/SebastianoMeneghin)


## The Project
The project can be divided into **four main pipelines**:
- **Historical data collection and preparation (historical feature pipeline)**: the features on weather condition and flight information have been extracted from [different source](#Results), such as *API Vendors* and *institutional OpenData archives*. Then, data have been processed, studied and uniformed to be used *for training*. The data are saved remotely in [Hopsworks Feature Store](https://www.hopsworks.ai/).
- **Daily data collection (real-time feature pipeline)**: everyday new data are extracted from various sources and processed, accordingly to the *standards and rules* set by the historical data, to be used for data augmentation and quotidian model training.
- **Model training and evaluation (training pipeline)**: a new model is trained everyday, with a bigger amount of data, thanks to the new data gathered in the second pipeline and to daily scripts running remotely on [Modal](https://modal.com/). The model is then saved in Hopsworks Model Registry.
- **Batch prediction and result visualisation (inference pipeline)**: everyday at midnight the daily forecast and flight schedules for the two following days are collected by a remote script. The model is accessed from the remote storage and delay predictions are made. The prediction results can be accessed through a [*user interface*](https://huggingface.co/spaces/SebastianoMeneghin/flight_delay) hosted on [*HuggingFace*](https://huggingface.co/) 🤗 created with [*Gradio*](https://www.gradio.app/). This application allows to use and see the model's functionalities.


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
