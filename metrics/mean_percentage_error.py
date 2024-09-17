import numpy as np

def mean_percentage_error(y_true, y_pred):
    percentage_errors = ((y_true - y_pred) / y_true) * 100
    
    mpe = np.mean(percentage_errors)
    return mpe

y_true = np.array([100, 200, 300, 400])
y_pred = np.array([110, 190, 280, 420])

mpe_value = mean_percentage_error(y_true, y_pred)
print(f"Mean Percentage Error (MPE): {mpe_value:.2f}%")
