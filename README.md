# Explore-TimeGPT-Tabula9-and-Relational-Deep-Learning


# Project Overview
This repository contains comprehensive Jupyter Notebooks (Colabs) and detailed documentation for a series of experiments and demonstrations involving TimeGPT, Tabular data modeling, and Relational Deep Learning (RDL) using RelBench. Each section showcases code walkthroughs, results, and video presentations explaining the code, outputs, and insights.

## Table of Contents
1. [TimeGPT Demonstrations](#timegpt-demonstrations)
   - [Multivariate Forecasting](#multivariate-forecasting)
   - [Long-Horizon Forecasting](#long-horizon-forecasting)
   - [Fine-Tuning with Custom Data](#fine-tuning-with-custom-data)
   - [Anomaly Detection](#anomaly-detection)
   - [Energy Forecasting](#energy-forecasting)
   - [Bitcoin Forecasting](#bitcoin-forecasting)
2. [Tabular Demonstrations](#tabular-demonstrations)
   - [Synthetic Data Generation](#synthetic-data-generation)
   - [Zero-Shot Inference](#zero-shot-inference)
3. [Relational Deep Learning (RDL) with RelBench](#rdl-with-relbench)
   - [Training a GNN Model](#training-a-gnn-model)
4. [Artifacts and Video Presentations](#artifacts-and-video-presentations)

---

## TimeGPT Demonstrations
### a) Multivariate Forecasting
**Colab Overview**:
This notebook demonstrates multivariate time series forecasting using TimeGPT with example data.

**Code Explanation**:
- Imports necessary libraries for time series processing.
- Loads sample multivariate data.
- Uses TimeGPT for forecasting multiple time series simultaneously.

**Resources**:
[TimeGPT Multivariate Forecasting Tutorial](https://docs.nixtla.io/docs/tutorials-multiple_series_forecasting)

### b) Long-Horizon Forecasting
**Colab Overview**:
This notebook showcases how to forecast over an extended time horizon using TimeGPT.

**Code Explanation**:
- Prepares long-horizon data.
- Configures TimeGPT for long-term forecasting and evaluates model accuracy.

### c) Fine-Tuning with Custom Data
**Colab Overview**:
Fine-tunes the TimeGPT model with custom time series data to improve performance on specific use cases.

**Code Explanation**:
- Custom data preprocessing.
- Training TimeGPT with new parameters.
- Performance comparison before and after fine-tuning.

**Resources**:
[TimeGPT Fine-Tuning Guide](https://docs.nixtla.io/docs/tutorials-fine_tuning)

### d) Anomaly Detection
**Colab Overview**:
Detects anomalies in time series data using TimeGPT.

**Code Explanation**:
- Data loading and pre-processing.
- Configuring anomaly detection parameters.
- Visualization of detected anomalies.

**Resources**:
[TimeGPT Anomaly Detection Tutorial](https://docs.nixtla.io/docs/tutorials-anomaly_detection)

### e) Energy Forecasting
**Colab Overview**:
Utilizes TimeGPT to forecast energy demand with sample data.

**Code Explanation**:
- Integrates energy data into the TimeGPT model.
- Evaluates forecasting performance specific to energy use cases.

**Resources**:
[Energy Forecasting Use Case](https://docs.nixtla.io/docs/use-cases-forecasting_energy_demand)

### f) Bitcoin Forecasting
**Colab Overview**:
Forecasts Bitcoin prices using TimeGPT with financial time series data.

**Code Explanation**:
- Loads cryptocurrency data.
- Configures TimeGPT for financial forecasting.
- Presents forecast results.

**Resources**:
[Bitcoin Price Prediction Use Case](https://docs.nixtla.io/docs/use-cases-bitcoin_price_prediction)

---

## Tabular Demonstrations
### a) Synthetic Data Generation
**Colab Overview**:
Generates synthetic data to augment or replace real data sets using Tabula.

**Code Explanation**:
- Loads a sample real dataset.
- Applies Tabula to create synthetic data.
- Visualizes and validates the generated data.

**Resources**:
[Synthetic Data Generation Example](https://github.com/zhao-zilong/Tabula/blob/main/Tabula_on_insurance_dataset.ipynb)

### b) Zero-Shot Inference
**Colab Overview**:
Demonstrates how to use Tabula for zero-shot inference.

**Code Explanation**:
- Imports pre-trained Tabula models.
- Runs inference without prior training.
- Displays results and metrics.

**Resources**:
[Zero-Shot Inference Example](https://github.com/mlfoundations/rtfm/blob/main/notebooks/inference.ipynb)

---

## Relational Deep Learning (RDL) with RelBench
### a) Training a GNN Model
**Colab Overview**:
Trains a Graph Neural Network (GNN) using RelBench for tabular prediction tasks and evaluates model performance.

**Code Explanation**:
- Loads and processes tabular data.
- Configures a GNN-based model with RelBench.
- Trains the model and shows evaluation metrics.

**Resources**:
[RelBench Tutorial](https://relbench.stanford.edu/start/)  
[Training Model Colab](https://colab.research.google.com/github/snap-stanford/relbench/blob/main/tutorials/train_model.ipynb)



