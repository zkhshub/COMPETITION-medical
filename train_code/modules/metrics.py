import torchmetrics

def get_metric(metric_name):
    if metric_name == 'MAE':
        return torchmetrics.MeanAbsoluteError()
    
    elif metric_name == 'RMSE':
        return torchmetrics.MeanSquaredError(squared=False)
    
    elif metric_name == 'MAPE':
        return torchmetrics.MeanAbsolutePercentageError()
    
    else:
        raise NotImplementedError

