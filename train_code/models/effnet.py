from torch import nn
import timm
import torch

class EffNet(nn.Module):
    def __init__(self, **kwargs):
        super(EffNet, self).__init__()
        self.model = timm.create_model(model_name=kwargs['encoder'], 
                                       pretrained=kwargs['pretrained'],
                                       num_classes=kwargs['n_outputs']
                                       )
        
        hdim = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.meta_process = nn.Sequential(nn.Linear(11, 32), nn.BatchNorm1d(32))
        self.regressor = nn.Sequential(
            nn.Linear(in_features = hdim + 32, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=kwargs['n_outputs'])
        )

    def forward(self, x, meta_info):
        img_features = self.model(x)
        meta_features = self.meta_process(meta_info)
        features = torch.cat((img_features, meta_features), dim=1)
        output = self.regressor(features)
        return output