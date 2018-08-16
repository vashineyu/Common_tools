from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, accuracy_score
import pandas_ml as pdml
import pandas as pd
import numpy as np
import csv

def create_report_binary(y_true, y_pred):
    """
    Create quick summary,
    Args
    - Input
    y_true: ground-truth (n_batch x n_classes), one-hot format
    y_pred: prediction (n_batch x n_classes), one-hot format
    - Return
    res: pandas table of prediction/ground-truth/status (fp/fn)
    conf: pdml confusion matrix table
    auc: auc score
    
    EXAMPLE of Usage
    final_result, eval_metric_table, auc = create_report(y_val, val_pred)
    eval_metric_table.stats()
    """
    res = pd.DataFrame({'y_true':y_true.argmax(axis = 1), 
                        'y_pred':y_pred.argmax(axis = 1)})
    
    res['status'] = res['y_pred'] - res['y_true'] # 1: fp, -1: fn
    
    auc = roc_auc_score(y_true=res.y_true, 
                        y_score=res.y_pred)
    
    conf = pdml.ConfusionMatrix(y_true=res.y_true, 
                                y_pred=res.y_pred)
    
    return res, conf, auc

def write_result_to_csv(pdml_table, file_name):
    with open(file_name, 'w') as f:
        for i in pdml_table.stats().keys():
            this_line = i + "," + str(pdml_table.stats()[i]) + '\n'
            f.writelines(this_line)
    print("Successfully write result to %s" % file_name)
    return True
    
    