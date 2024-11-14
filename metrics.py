import numpy as np

def average_accuracy(result_matrix, t):
    """
    Calculate the average accuracy (A_t) for the t-th task.
    
    Parameters:
    - result_matrix (numpy.ndarray): A matrix where result_matrix[i][j] represents a_{i,j}.
    - t (int): The index of the task for which to calculate the average accuracy
    
    Returns:
    - float: The average accuracy for the t-th task.
    """
    return np.mean(result_matrix[t, :t+1])

def average_forgetting(result_matrix, t):
    """
    Calculate the average forgetting (F_t) for the t-th task.
    
    Parameters:
    - result_matrix (numpy.ndarray): A matrix where result_matrix[i][j] represents a_{i,j}.
    - t (int): The index of the task for which to calculate the average forgetting
    
    Returns:
    - float: The average forgetting for the t-th task.
    """
    if t == 0:
        return 0
    forgetting_sum = 0
    for i in range(t):
        max_performance_drop = max(result_matrix[j, i] - result_matrix[t, i] for j in range(i, t))
        forgetting_sum += max_performance_drop
    return forgetting_sum / t

def average_incremental_accuracy(result_matrix):
    """
    Calculate the average incremental accuracy (Ā_T) up to task T.
    
    Parameters:
    - result_matrix (numpy.ndarray): A matrix where result_matrix[i][j] represents a_{i,j}.
    
    Returns:
    - float: The average incremental accuracy up to task T.
    """
    # return np.mean([average_accuracy(result_matrix, t) for t in range(1, result_matrix.shape[0])])
    # I think we need to add the accuracy of first task
    return np.mean([average_accuracy(result_matrix, t) for t in range(0, result_matrix.shape[0])]) 

def average_incremental_forgetting(result_matrix):
    """
    Calculate the average incremental forgetting (F̄_T) up to task T.
    
    Parameters:
    - result_matrix (numpy.ndarray): A matrix where result_matrix[i][j] represents a_{i,j}.
    
    Returns:
    - float: The average incremental forgetting up to task T.
    """
    return np.mean([average_forgetting(result_matrix, t) for t in range(1, result_matrix.shape[0])])

def learning_accuracy(result_matrix):
    """
    Calculate the average learning accuracy from the diagonal elements of the result matrix.
    
    Parameters:
    - result_matrix (numpy.ndarray): A matrix where result_matrix[i][i] represents a_{i,j}.
    
    Returns:
    - float: The average learning accuracy based on the diagonal elements of the result matrix.
    """
    return np.mean(np.diag(result_matrix))