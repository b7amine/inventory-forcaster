# Inventory Forecasting Project POC

### Modeling Framework Design Choices and Trade-offs

Given the identified short-term and long-term patterns in the time series data, and in the absence of business constraints to work backward from, I established a modeling framework:
- **Data Aggregation**: Aggregate data weekly to reduce noise and reinforce patterns.
- **Observation Updates**: Update observations every 4 weeks, predicting the next 4 weeks based on the previous 4 weeks.
- **Seq2Seq Model**: Utilize an encoder-decoder architecture with an attention mechanism.
- **Uncertainty Measurement**: Using dropout to yield 95% confidence bands and normalized entropy of the prediction distribution.
- **Evaluation Metric**: Use MAPE quantiles to handle outliers and provide detailed insights.

### Results and Insights

#### Seq2Seq Model
- **Error Quantiles**: 
  - Min: 1
  - 25%: 38
  - 50%: 47
  - 85%: 75
  - 95%: 100
  - Max: 513

#### Baseline Model **SARIMA**
   - **Method**: Fit on the first 24 observations, then predict the next 4 steps using a sliding window, appending ground truth.
   - **MAPE Quantiles**: 
     - Min: 18.14
     - 25%: 54.63
     - 50%: 76.06
     - 85%: 137.59
     - 95%: 196.25
     - Max: --

Among various model configurations, the best Seq2Seq model had:
- `hidden_dim` = 128
- 2 layers
- 0.5 dropout
- Attention and incremental encoder context updates
- Inclusion of `current_level`
- Previous observations and their corresponding positional encoding (cos,sin week_of_year) , allowing non consecutive observations to be fed + other benefits


### Key Findings (results_and_insights.ipynb)

- The embedding space highlighted interesting clusters, indicating the model had learned some useful information, but there is room for further improvement.
- Attention analysis revealed patterns that could be linked to seasonality and the importance of recent steps in predictions.
- The Seq2Seq model showed potential but requires more tuning and exploration of other baselines like Temporal Convolution Networks and LagLlama.


## Project Structure

After unzipping `datasets.zip`, the repository is organized as follows:

- **datasets/**: Contains the initial dataset and additional datasets generated during the process.
- **notebooks/**:
  - **exploratory_analysis/**: Data cleaning, feature selection, and initial assumptions.
  - **modelling_experiments/**: Model training and hyperparameter tuning.
  - **baselines_benchmarks/**: Evaluation of baseline models and benchmarks.
  - **results_and_insights/**: Analysis of model interpretability, embedding space, attention mechanisms, and uncertainty.
- **models/**: Stores different model instances created during training.
- **predict/**: Prediction module.
  - **samples/**: Individual samples for running predictions.
  - **output/**: Stores prediction results for samples.
  - **predict.py**: Iterates over samples in the `samples/` directory and generates predictions.

## How to Use

To use this project:
1. Unzip `datasets.zip`.
2. Navigate through the `notebooks/` directory to explore the analysis and modeling steps.
3. Use the `predict/` module to generate predictions on new samples.

Ensure you have the necessary dependencies installed, including Python 3.11, Plotly, and nbformat >= 4.2.0. More details in requirements.txt

## Conclusion

With further tuning and exploration, the models developed here have the potential to provide valuable insights and more accurate forecasts for inventory management.
