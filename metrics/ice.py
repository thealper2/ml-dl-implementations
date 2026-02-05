def individual_conditional_expectation_(model, X, feature_index, grid):
    n_samples = X.shape[0]
    ice_values = np.zeros((n_samples, len(grid)))
    
    for i in range(n_samples):
        X_temp = np.repeat(X[i:i+1], len(grid), axis=0)
        X_temp[:, feature_index] = grid
        ice_values[i] = model(X_temp)
        
    return ice_values
