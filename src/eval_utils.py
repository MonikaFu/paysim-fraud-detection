import pandas as pd
from sklearn import metrics

def performance_assessment(predictions_df, output_feature='isFraud', 
                           prediction_feature='predictions',
                           rounded=True):
    
    AUC_ROC = metrics.roc_auc_score(predictions_df[output_feature], predictions_df[prediction_feature])
    AP = metrics.average_precision_score(predictions_df[output_feature], predictions_df[prediction_feature])
    
    performances = pd.DataFrame([[AUC_ROC, AP]], 
                           columns=['AUC ROC','Average precision'])
        
    if rounded:
        performances = performances.round(3)
    
    return performances