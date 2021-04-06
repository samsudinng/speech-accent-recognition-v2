import torchvision
import torch.nn as nn

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class AlexNetGAP(nn.Module):
   
    def __init__(self,num_classes=8, in_ch=3, pretrained=True):
        super(AlexNetGAP, self).__init__()

        model = torchvision.models.alexnet(pretrained=pretrained)
        self.features = model.features
        self.avgpool  = model.avgpool
        
        #Global average pooling layer
        self.classifier = nn.Sequential(
                                        nn.Conv2d(256, num_classes, kernel_size=(1,1)),
                                        nn.AvgPool2d(6)
                                        )
        init_layer(self.classifier[0])


        if in_ch != 3:
            self.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
            init_layer(self.features[0])
        
    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        out = self.classifier(x).squeeze(-1).squeeze(-1)

        return out


class VGG16GAP(nn.Module):

    def __init__(self,num_classes=8, in_ch=3, pretrained=True):
        super(VGG16GAP, self).__init__()

        model = torchvision.models.vgg16(pretrained=pretrained)
        self.features = model.features
        self.avgpool  = model.avgpool

        #Global average pooling layer
        self.classifier = nn.Sequential(
                                        nn.Conv2d(512, num_classes, kernel_size=(1,1)),
                                        nn.AvgPool2d(7)
                                        )
        init_layer(self.classifier[0])


        if in_ch != 3:
            self.features[0] = nn.Conv2d(in_ch, 64, kernel_size=3, padding=1)
            init_layer(self.features[0])


    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        out = self.classifier(x).squeeze(-1).squeeze(-1)

        return out



class VGG16BnGAP(nn.Module):

    def __init__(self,num_classes=8, in_ch=3, pretrained=True):
        super(VGG16BnGAP, self).__init__()

        model = torchvision.models.vgg16_bn(pretrained=pretrained)
        self.features = model.features
        self.avgpool  = model.avgpool

        #Global average pooling layer
        self.classifier = nn.Sequential(
                                        nn.Conv2d(512, num_classes, kernel_size=(1,1)),
                                        nn.AvgPool2d(7)
                                        )
        init_layer(self.classifier[0])


        if in_ch != 3:
            self.features[0] = nn.Conv2d(in_ch, 64, kernel_size=3, padding=1)
            init_layer(self.features[0])


    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        out = self.classifier(x).squeeze(-1).squeeze(-1)

        return out


class Resnet34Var(nn.Module):

    def __init__(self,num_classes=8, in_ch=3, pretrained=True):
        super(Resnet34Var, self).__init__()
            
        model = torchvision.models.resnet34(pretrained=pretrained)
        model.fc=nn.Linear(in_features=512, out_features=num_classes, bias=True)
        init_layer(model.fc)
                                    
        self.model= model
                                                    
    def forward(self, x):
                                                        
        return self.model(x)



class Resnet18Var(nn.Module):

    def __init__(self,num_classes=8, in_ch=3, pretrained=True):
        super(Resnet18Var, self).__init__()
            
        model = torchvision.models.resnet18(pretrained=pretrained)
        model.fc=nn.Linear(in_features=512, out_features=num_classes, bias=True)
        init_layer(model.fc)
                                    
        self.model= model
                                                    
    def forward(self, x):
                                                        
        return self.model(x)


class Resnet50Var(nn.Module):

    def __init__(self,num_classes=8, in_ch=3, pretrained=True):
        super(Resnet50Var, self).__init__()
            
        model = torchvision.models.resnet50(pretrained=pretrained)
        model.fc=nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        init_layer(model.fc)
                                    
        self.model= model
                                                    
    def forward(self, x):
                                                        
        return self.model(x)


                                                        
