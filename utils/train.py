import math
import torch
import torch.nn as nn
import torch.optim as optim
import os
import shutil
from shutil import copyfile
from utils.train_util import *
from utils.utils import *



def train(model, train_dataloader, valid_dataloader, criterion, num_epochs, clf=False, optimizer=None, scheduler=None, device='cuda', model_name='Predictor', tag=""):
    init_change_patience = 5
    
    if os.path.exists(f"./model/{model_name}/{tag}_ckpt_best.pt"):
        print("Best Model Exist")
        model.load_state_dict(torch.load(f"./model/{model_name}/{tag}_ckpt_best.pt"))
        return model, None, None
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-8)
    if scheduler is None:
        scheduler =  CosineAnnealingWarmUpRestarts(optimizer, T_0=20, T_mult=1, eta_max=0.1, T_up=60, gamma=0.5)

    es = EarlyStopping(patience=10, delta=0, mode='min', verbose=True)

    es_init = EarlyStopping(patience=init_change_patience, delta=0, mode='min', verbose=False)
    total_train_loss = []
    total_valid_loss = []

    model.to(device)
    createFolder(f'./model/{model_name}')

    for epoch in range(num_epochs):
        train_loss = []
        for data, labels in train_dataloader:
            model.train()
            optimizer.zero_grad()
            
            labels = labels.to(device)
            inputs = data.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
    
            loss.backward()
            optimizer.step()
    
            train_loss.append(loss.item())
        
        print(f"Epoch {epoch} Loss: ", np.mean(train_loss))
        total_train_loss.append(np.mean(train_loss))
        
        scheduler.step()
        torch.save(model.state_dict(), f"./model/{model_name}/{tag}_ckpt_{epoch}.pt")
        
        model.eval()
        with torch.no_grad():
            valid_loss = []
            acc_list = []
            for data, labels in valid_dataloader:
                labels = labels.to(device)
                inputs = data.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss.append(loss.item())
                if clf:
                    pred = torch.flatten(torch.argmax(outputs, dim=1))
                    true = torch.flatten(labels)
                    acc_list.extend(list((pred==true).detach().cpu().numpy()))
                    
            if len(acc_list)!=0:
                print(f"Epoch {epoch} Eval ACC: ", sum(acc_list)/len(acc_list))
                        
            print(f"Epoch {epoch} Eval Loss: ", np.mean(valid_loss))
            total_valid_loss.append(np.mean(valid_loss))
            es(np.mean(valid_loss))
            es_init(np.mean(valid_loss))
            
            
        if es.early_stop:
            break
        # if es_init.early_stop:
        #     print("Stuck in initial point - Re Train")
        #     torch.cuda.empty_cache()
        #     model, total_train_loss, total_valid_loss = train(model, train_dataloader, valid_dataloader,
        #                                           criterion, num_epochs=num_epochs, device=device,  
        #                                           model_name=model_name,
        #                                           tag=tag)
        #     return model, total_train_loss, total_valid_loss
            
    best_model_epoch = np.argmin(total_valid_loss)
    copyfile(f"./model/{model_name}/{tag}_ckpt_{epoch}.pt", f"./model/{model_name}/{tag}_ckpt_best.pt")
    
    return model, total_train_loss, total_valid_loss