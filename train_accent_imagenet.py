import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_optimizer as optim2
import pandas as pd
import os
import torchaudio.transforms as audiotransforms
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from model_imagenet import AlexNetGAP, VGG16BnGAP, VGG19BnGAP, VGG16GAP, Resnet34Var, Resnet18Var, Resnet50Var
from dataloader import AccentImageTESTDataset
import os
import argparse
import sys
import json

label_to_accent=['US','UK','CH','IN','JP','KR','PT','RU']

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def fprint(filename, txt):
    with open(filename,"a") as fi:
        print(txt, file=fi)
    

### Classifier

def main(args):
    
    #read config file
    with open(args.config, 'r') as f:
        config = json.load(f)
 
    logfile = args.logdir + '/log.file'
    writer = SummaryWriter(log_dir=args.logdir)
    
    #Train/dev/test/metadata paths
    mpath = config['mpath']
    trainfpath=config['trainpath']
    devfpath=config['devpath']
    testfpath=config['testpath']
    
    #Hyper-param
    batchsize = config['batchsize']
    num_epoch = config['epochs']        
    
    #Model
    model = eval(config['model'])    

    #Device
    if torch.cuda.is_available():
        device=torch.device('cuda')
        fprint(logfile, 'use GPU')
    else: 
        device=torch.device('cpu')
        fprint(logfile, 'use CPU')
    
    #Load checkpoint if resuming
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    
    
    #Loss Function
    lossfunc = eval(config['loss'])
    
    #Optimizer
    optimizer = eval(config['optimizer'])
    scheduler = eval(config['scheduler'])
 
    
    #Load checkpoint/initialize    
    if args.resume is not None:
        epoch_start = checkpoint['epoch']+1
        epoch_resume= 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        torch.set_rng_state(checkpoint['torch_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        best_acc = checkpoint['best_acc']
    else:
        epoch_start = 0
        epoch_resume= 0
        best_acc = 0.0
    
    
    #Dataset
    p_augment = config['p_specaugment']
    
    if config['feature'] == 'spectrogram' :
        train_transforms = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.ToTensor(), 
                                            transforms.RandomChoice([
                                                transforms.RandomApply([audiotransforms.FrequencyMasking(freq_mask_param=50)], p=p_augment),
                                                transforms.RandomApply([audiotransforms.TimeMasking(time_mask_param=100)], p=p_augment)
                                            ]),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])

                                 ])
    
        test_transforms = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])

                           ])
    elif config['feature'] == 'wav2vec':
        train_transforms = transforms.Compose([
                                            transforms.ToTensor(), 
                                            #transforms.RandomChoice([
                                            #    transforms.RandomApply([audiotransforms.FrequencyMasking(freq_mask_param=50)], p=p_augment),
                                            #    transforms.RandomApply([audiotransforms.TimeMasking(time_mask_param=100)], p=p_augment)
                                            #]),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])

        ])
    
        test_transforms = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])
        ])


    
    
    train_ds = ImageFolder(root=trainfpath, transform=train_transforms) 
    dev_ds = AccentImageTESTDataset(mpath+'devset.csv', devfpath, transform=test_transforms)
    test_ds = AccentImageTESTDataset(mpath+'testset.csv', testfpath, transform=test_transforms)

    #Dataloader
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batchsize, num_workers=config['num_workers'], shuffle=True)
    dev_dataloader = torch.utils.data.DataLoader(dev_ds, batch_size=1, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)
    
    fprint(logfile,"\nDataloader")
    fprint(logfile,f"train: {len(train_dataloader)}")
    fprint(logfile,f"dev: {len(dev_dataloader)}")
    fprint(logfile,f"test: {len(test_dataloader)}")

    #Epoch
    for epoch in range(epoch_start,num_epoch):
    
        ###############    
        #### TRAIN ####
        ###############
        model.train()
        train_loss = 0.0
        correct = 0
        num_train = 0
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
    
            wavtensor, target = batch
            wavtensor = wavtensor.to(device)
            target = target.to(device)
    
            out = model(wavtensor)
            
            
            #loss
            loss = lossfunc(out, target)
            train_loss += loss.detach().item()
        
            #accuracy
            predicted = torch.argmax(torch.nn.functional.softmax(out.detach(),dim=1),dim=1)
            predicted_correct = (predicted == target).sum().item()
            correct += predicted_correct
            curr_batchsize = predicted.size()[0]
            num_train += curr_batchsize                    
            
            loss.backward()
    
            optimizer.step()
        
            writer.add_scalar('Train/loss', loss.detach().item(), epoch * len(train_dataloader) + i)
            writer.add_scalar('Train/accuracy', predicted_correct/curr_batchsize, epoch * len(train_dataloader) + i)
        
                 
        train_loss = train_loss/len(train_dataloader)
        train_loss = np.around(train_loss, decimals = 4)
        train_acc = correct/num_train
        train_acc = np.around(train_acc, decimals = 4)
    
        
        model.eval()
        with torch.no_grad():

            ############
            ### DEV ####
            ############
            dev_correct, running_correct = 0, 0
            dev_loss, running_loss = 0.0, 0.0
           
            for i, batch in enumerate(dev_dataloader):
                wavtensor, target = batch
                wavtensor = wavtensor.to(device)
                target = target.to(device)
                            
                out = model(wavtensor)
            
                loss = lossfunc(out, target)
                dev_loss += loss.item()
                running_loss += loss.item()
            
                predicted = torch.argmax(torch.nn.functional.softmax(out.detach(),dim=1),dim=1)                        
                predicted_correct = (predicted == target).sum().item()
                dev_correct += predicted_correct
                running_correct += predicted_correct
            
                if i % 128 == 127:
                    writer.add_scalar('Dev/loss', running_loss/128, epoch * len(dev_dataloader) + i)
                    writer.add_scalar('Dev/accuracy', running_correct/128, epoch * len(dev_dataloader) + i)
                    running_loss = 0.0
                    running_correct = 0
                
            dev_acc = dev_correct/dev_ds.num_utt
            dev_acc = np.around(dev_acc, decimals = 4)
            dev_loss = dev_loss/len(dev_dataloader)
            dev_loss = np.around(dev_loss, decimals = 4)
            
            scheduler.step(dev_acc)
            
            ############
            ### TEST ###
            ############
            test_correct = 0
            all_predicted, all_target = [],[]
            
            for i, batch in enumerate(test_dataloader):
                wavtensor, target = batch
                wavtensor = wavtensor.to(device)
                target = target.to(device)
                            
                out = model(wavtensor)
                
                predicted = torch.argmax(torch.nn.functional.softmax(out.detach(),dim=1),dim=1)                        
                predicted_correct = (predicted == target).sum().item()
                test_correct += predicted_correct
                
                all_predicted.extend(list(predicted.cpu().numpy()))
                all_target.extend(list(target.cpu().numpy()))

            test_acc = test_correct/test_ds.num_utt
            test_acc = np.around(test_acc, decimals = 4)
            
            writer.add_scalar('Accuracy/train',train_acc, epoch)
            writer.add_scalar('Accuracy/dev',dev_acc, epoch)
            writer.add_scalar('Accuracy/test',test_acc, epoch)
            
            #save the best model
            if dev_acc > best_acc:
                best_acc = dev_acc
                os.system(f"rm {args.logdir}/bestmodel*")
                torch.save(model.state_dict(),f'{args.logdir}/bestmodel_epoch{epoch}.pth' )        
    
        results = { 'Epoch'         :[epoch+1],
                    'Train loss'    :[train_loss],
                    'Dev loss'      :[dev_loss],
                    '|'             :['|'],
                    'Train acc.'    :[train_acc],
                    'Dev acc.'      :[dev_acc],
                    'Test acc.'     :[test_acc]
        }
    
        df = pd.DataFrame.from_dict(results).set_index('Epoch')     
        fprint(logfile,'\n'+'-'*40)
        fprint(logfile,f"Number of training segments: {num_train}")
        fprint(logfile,df)
        
        #confusion matrix
        cm = np.round(confusion_matrix(all_target,all_predicted,labels=list(range(len(label_to_accent))),normalize='true')*100,2)
        df=pd.DataFrame(cm)
        df.index = label_to_accent
        df.columns = label_to_accent
    
        fprint(logfile, '\n')
        fprint(logfile, df)
                    
        #save checkpoint
        checkpoint = {
                   'state_dict': model.state_dict(),
                   'optimizer' : optimizer.state_dict(),
                   'scheduler' : scheduler.state_dict(),
                   'epoch'     : epoch,
                   'torch_rng' : torch.get_rng_state(),
                   'numpy_rng' : np.random.get_state(),
                   'best_acc'  : best_acc
        }
    
        torch.save(checkpoint, f'{args.logdir}/checkpoint_spectrogram_epoch{epoch}.pth')
        
        if epoch > epoch_start+1 and epoch_resume == 0:
            #delete the previous checkpoint
            os.system(f"rm {args.logdir}/checkpoint_spectrogram_epoch{epoch-2}.pth")
        else:
            epoch_resume = 0

def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Spectrogram-based Accent Recognition V2')
    parser.add_argument('--logdir', default=None, type=str,
                                  help='directory to save checkpoints and results')
    parser.add_argument('--config', default=None, type=str,
                                  help='config file')
    parser.add_argument('--resume', default=None, type=str,
                              help='checkpoint file (.pth)')
    
    return parser.parse_args(argv)


if __name__ == '__main__':

    main(parse_arguments(sys.argv[1:]))
