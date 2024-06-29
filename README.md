# Implementing Shrinkage Risk Parity and Hierarchical Risk Parity Portfolio Models Utilizing Noise Reduction and Machine Learning Techniques

## Project Overview
This project focuses on implementing and comparing two advanced portfolio optimization methods: Hierarchical Risk Parity (HRP) and Shrinkage Risk Parity (SRP). The primary goal is to develop and evaluate these models for asset allocation using historical ETF data, incorporating advanced noise reduction and machine learning techniques.

## Methods
### Hierarchical Risk Parity (HRP)
HRP uses hierarchical clustering to construct a diversified portfolio by avoiding concentration in correlated assets.

### Shrinkage Risk Parity (SRP)
SRP combines risk parity with the Ledoit-Wolf shrinkage technique to improve the stability of the covariance matrix estimation, reducing noise and enhancing performance.

### Ledoit-Wolf Shrinkage
The Ledoit-Wolf shrinkage method aims to improve the estimation of the covariance matrix by addressing the instability of the sample covariance matrix, especially in high-dimensional settings. This is crucial for accurate risk assessment and effective portfolio optimization.


## Data
The project uses ETF data (`etf_data.csv`) containing historical prices and other relevant financial information from January 1, 2005, to May 31, 2024.

## Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/fosh-pse/CPF_Final-Project_Seungeun-Park.git
   cd CPF_Final-Project_Seungeun-Park
   ```

2. **Run Jupyter Notebook**:
   Open `HRP_SRP_Comparison.ipynb` in Jupyter Notebook or Google Colab to execute the code and see the results.

## Results
The results section includes a detailed comparison of HRP and SRP models based on their performance metrics from backtesting. This includes cumulative returns, Sharpe ratios, and risk contributions of individual assets within the portfolio.

## Conclusion
This project demonstrates the effectiveness of HRP and SRP models in portfolio optimization. The detailed analysis and backtesting results provide insights into the strengths and limitations of each method. Both methodologies offer substantial benefits in terms of risk management and volatility reduction, making them valuable tools for enhancing portfolio stability and performance.

## Authors
Seungeun Park

For more details, visit the [project repository](https://github.com/fosh-pse/CPF_Final-Project_Seungeun-Park) or the [Google Colab notebook](https://colab.research.google.com/drive/1trekj0sjzcWNIqxPw6X_awDXmudX3CHh#scrollTo=579cc4ac-887a-49aa-9cf1-968cb4dc0055).
