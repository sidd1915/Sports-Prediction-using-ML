# IPL Match Winner Predictor

## Overview

This project is designed to predict the winner of an IPL match based on several input parameters. The application uses a machine learning model to make predictions and is built with Flask for the backend.
Dataset can be found [here](https://www.kaggle.com/datasets/rajsengo/indian-premier-league-ipl-all-seasons?select=points_table.csv).

## Features

- **Input Parameters:**
  - Home Team
  - Away Team
  - Venue
  - Toss Won By
  - Decision (Bat/Field)
  - Net Run Rate (NRR)

- **Output:**
  - Predicted Winner of the match

## Technologies Used

- **Backend:** Flask
- **Machine Learning Model:** XGBoost
- **Data Handling:** Pandas

## Setup

### Prerequisites

- Python 
- Flask
- Pandas
- Scikit-learn
- XGBoost

### Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>

2. **Create and Activate a Virtual Environment**:
   ```bash
   python3 -m venv venv
   venv\Scripts\activate

3. **Run the Application**:
   ```bash
   python app.py**:

The application will be available at http://127.0.0.1:5000.

### Usage
![image](https://github.com/user-attachments/assets/860f54fb-39ed-4f68-809a-83662ef753a7)
![image](https://github.com/user-attachments/assets/b88d8d8b-bf89-42b9-901d-af5ece9db65e)
![image](https://github.com/user-attachments/assets/334b74eb-e925-490c-9a47-1a665aab136f)

### License
This project is licensed under the MIT License. See the LICENSE file for details.
