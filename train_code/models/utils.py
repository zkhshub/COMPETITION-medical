from models.effnet import EffNet

def get_model(model_name:str, model_args:dict):
    if model_name == 'effnet':
        return EffNet(**model_args)
    else:
        raise NotImplementedError(f'{model_name} is not implemented')
    
if __name__ == '__main__':
    pass