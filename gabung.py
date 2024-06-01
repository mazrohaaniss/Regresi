import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Path to the CSV file
file_path = r'D:\SEMESTER 4\metode numerik\Tugas 3\Student_Performance.csv'

# Read the CSV file with delimiter ';'
data = pd.read_csv(file_path, delimiter=';')

# Check the columns in the dataframe to ensure 'NL' and 'NT' columns are present
print("Columns in data:", data.columns)

# Assuming columns in CSV are 'NL' for Number of Exercises and 'NT' for Test Scores
TB = data['TB']
NT = data['NT']

# Linear Regression Model
X = TB.values.reshape(-1, 1)
y = NT.values
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)
rms_error_linear = np.sqrt(np.mean((y - y_pred_linear) ** 2))
print("RMS Error (Linear Regression):", rms_error_linear)

# Exponential Regression Model
log_y = np.log(y)
exponential_model = LinearRegression()
exponential_model.fit(X, log_y)
log_y_pred = exponential_model.predict(X)
y_pred_exponential = np.exp(log_y_pred)
rms_error_exponential = np.sqrt(np.mean((y - y_pred_exponential) ** 2))
print("RMS Error (Exponential Regression):", rms_error_exponential)

# Plotting the graph
plt.scatter(TB, NT, label='Original Data')
plt.plot(TB, y_pred_linear, color='red', label='Linear Regression')
plt.plot(TB, y_pred_exponential, color='green', label='Exponential Regression')
plt.xlabel('Waktu Belajar (TB)')
plt.ylabel('Nilai Ujian (NT)')
plt.legend()
plt.title('Hubungan antara Waktu Belajar dan Nilai Ujian (Linear and Exponential Regression)')
plt.show()
