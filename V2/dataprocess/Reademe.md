
# Data Split Strategies

This directory contains two different data splitting strategies used for constructing samples in experiments.

## 1. sessionSplit

 `sessionSplit` folder uses a **session-based data splitting strategy**:

- User interactions within a **continuous time period (a session)** are treated as one data sample;
- Each session corresponds to a single data instance;

## 2. LLM4POI_data

You can directly adopt the preprocessed dataset from **LLM4POI**:  
https://github.com/neolifer/LLM4POI/tree/main

To reproduce our data pipeline, please follow the steps below:

1. **Download the dataset**  
   Download the dataset from the LLM4POI repository.

2. **Generate POI metadata**  
   Run the following notebook to obtain `poi_info.csv`: `LLM4POI_data/dataprocess.ipynb`

3. **Generate Semantic IDs (SIDs)**  
Execute the codebook quantization module to obtain SID representations for POIs.

4. **Prepare LLM training data**  
Run the following notebook to construct the final dataset for LLM fine-tuning: `LLM4POI_data/llm_dataprocess.ipynb`

## Historical Data Concatenation

Both data splitting strategies apply historical data concatenation:

- The historical sequence length is set to **50** in the experiments;
- Each data sample consists of the current interactions concatenated with the **most recent 50 historical records**.
