# Workshop Data Science & AI 2022-2023

In this repo I have saved all the NB used during the workshop in July22 (6 Rome, 7 Milan)
and in all following events (ACN, Sisal, Tods, ...)

## Features and capabilities explored

* How to use a **custom conda** environment
* how-to **read data from ADWH**
* Tuning hyper-parameters using **ADSTuner**
* How to save an entire **Sklearn** pipeline to **Model Catalog**
* usage of the new ADS framework for **Model Serialization**
* Advanced usage of **Model Catalog**
* **Deploy** a Model directly using ADS (code)
* Usage of **ADS Feature Types**
* How-to use **Apache Spark** in a NB session
* How-to use the OCI Data Science -> **DataFlow** integration
* new AutoMLX package

## Apache Spark 

Apache Spark is a modern and powerful framework for distributed computing.
Using Apache Spark you can process, in a reasonable amount of time, very large datasets stored on Object Storage, distributing the computations on a cluster.
In the provided Notebooks we will see:
* how to use Spark and Spark SQL directly in a Notebook Session, using a dedicated Conda env
* how to develop and run a Spark JOB using OCI DataFlow.

## Feature Types

ADS has a very powerful and interesting framework to specify and check **Feature Types**.
In the Notebook [Complete Example with Feature Types](https://github.com/luigisaetta/workshop-july-2022/blob/main/complete_example_model_creation_deployment_feature_types.ipynb) 
I have used Feature Types to register which features are categorical and which continuous.



