
# CPF_Final-Project_Seungeun-Park

## Overview
Accurate prediction of asset price movements is crucial for effective trading strategies. This project involves implementing portfolio optimization techniques using Python. The main techniques covered are Hierarchical Risk Parity (HRP) and Shrinkage Risk Parity (SRP).

### Project Description
This project explores the implementation of two versions of Risk Parity portfolios using advanced noise reduction and machine learning techniques. The focus is on comparing the performance of the Shrinkage Risk Parity Model and the Hierarchical Risk Parity Model, utilizing a CSV file for data acquisition. The models' performances are evaluated through extensive back-testing on historical ETF data. The objective is to optimize asset allocation based on risk contributions, aiming to enhance portfolio returns and examine the impact of noise reduction methods on the accuracy and stability of the portfolios.

## Files
- `CPF_Final_Project_Seungeun_Park(June_2024)_20240628.ipynb`: The Jupyter Notebook containing all code and analysis.
- `etf_data.csv`: The data file containing historical ETF prices.

## Instructions

### Using GitHub
1. Ensure the `etf_data.csv` file is uploaded to the same GitHub repository as the Jupyter Notebook.
2. Open the Jupyter Notebook in Google Colab.
3. The code to read the CSV file from GitHub is already included in the notebook:
   ```python
   import pandas as pd
   import io
   import requests

   url = 'https://raw.githubusercontent.com/fosh-pse/CPF_Final-Project_Seungeun-Park/main/etf_data.csv'
   response = requests.get(url)
   df = pd.read_csv(io.StringIO(response.text))
   df = load_etf_price_data(df)
   ```

## Dependencies
These packages are required to run the notebook and can be installed using pip:
```bash
pip install numpy pandas matplotlib seaborn scipy requests
```

### Additional Packages
Ensure the following additional packages are imported in the notebook:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
import os
import requests
import io
```
