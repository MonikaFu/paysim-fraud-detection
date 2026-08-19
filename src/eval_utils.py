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

def performance_assessment_model_collection(fitted_models_and_predictions_dictionary, 
                                            transactions_df, 
                                            type_set='test',
                                            output_feature='isFraud'):
    performances=pd.DataFrame() 
    
    for classifier_name, model_and_predictions in fitted_models_and_predictions_dictionary.items():
    
        predictions_df=transactions_df
            
        predictions_df['predictions']=model_and_predictions['predictions_'+type_set]
        
        performances_model=performance_assessment(predictions_df, output_feature=output_feature, 
                                                   prediction_feature='predictions')
        performances_model['training_execution_time']=model_and_predictions['training_execution_time']
        performances_model['prediction_execution_time']=model_and_predictions['prediction_execution_time']

        performances_model.index=[classifier_name]
        performances=pd.concat([performances, performances_model])
        
    return performances