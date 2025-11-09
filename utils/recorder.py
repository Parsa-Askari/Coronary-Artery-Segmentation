import numpy as np
import json
import os
import matplotlib.pyplot as plt
###IE###
###SS###
class HistoryRecorder:
    def __init__(self,class_maps,losses_keys,class_count = 25):

        self.history = {
            "train":{},
            "valid":{}
        }
        for key in losses_keys:
            self.history["train"][key] = [[]]
            self.history["valid"][key] = [[]]
        self.losses_keys = losses_keys
        self.metric_history={}
        self.class_maps =class_maps
        for part in self.history :
            self.metric_history[part]={"dice":{},"precision":{},"recall":{}}
            for i in range(1,class_count+1):
                self.metric_history[part]["dice"][i]=[]
                self.metric_history[part]["recall"][i]=[]
                self.metric_history[part]["precision"][i]=[]
                
        self.metric_avg_list = {
            "train":{"dice":[],"precision":[],"recall":[]},
            "valid":{"dice":[],"precision":[],"recall":[]}
        }

        self.class_count = class_count 
        
    def add_losses(self,part,loss_dict):
        for loss_name,loss in loss_dict.items():
            self.history[part][loss_name][-1] += [loss]
        
    def add_metrics(self,dice,precision,recall,part):
        dice = dice[1:]
        precision = precision[1:]
        recall = recall[1:]
        for i in range(self.class_count):
            d = dice[i]
            r = recall[i]
            p = precision[i]
            self.metric_history[part]["dice"][i+1].append(d)
            self.metric_history[part]["recall"][i+1].append(r)
            self.metric_history[part]["precision"][i+1].append(p)
    def avg_losses(self,part):
        for key in self.history[part]:
            self.history[part][key][-1] = np.mean(self.history[part][key][-1])
            self.history[part][key].append([])
            
    def print_loss_report(self,part,epoch,avg_first=True):
        if(avg_first):
            self.avg_losses(part)
            
        report = f"{part} ==> epcoh ({epoch})\n"
        co=0
        for loss_name in self.losses_keys:
            loss_list = self.history[part][loss_name]

            loss = loss_list[-2]
            report += f"{loss_name} : {loss}"
            if((co+1)%3==0):
                report += "\n"
            else:
                report+=" - "
            co+=1  
                 
        print(report)
    def print_metrics_report(self,part,epoch,class_wise=False):
        
        report_temp = f"{part} avg metrics for epoch {epoch} :\n"
        report_class_wise_temp = ""
        avg_dice = 0
        avg_precision = 0
        avg_recall = 0
        for index , c in self.class_maps.items():
            dice = self.metric_history[part]["dice"][index][-1]
            precision = self.metric_history[part]["precision"][index][-1]
            recall = self.metric_history[part]["recall"][index][-1]
            avg_dice += dice
            avg_precision += precision
            avg_recall += recall
            if(class_wise):
                report_class_wise_temp += f"{c} => dice : {dice} p : {precision} , r : {recall}\n"
        
        avg_dice = avg_dice/self.class_count
        avg_precision = avg_precision/self.class_count
        avg_recall = avg_recall/self.class_count

        self.metric_avg_list[part]["dice"]+=[avg_dice]
        self.metric_avg_list[part]["precision"]+=[avg_precision]
        self.metric_avg_list[part]["recall"]+=[avg_recall]
        
        report_temp+=f"avg dice : {avg_dice} - avg precision : {avg_precision} - avg recall : {avg_recall}"

        if(class_wise):
            report_temp = report_temp + "\n" +report_class_wise_temp[:-1] #removing last \n
        print(report_temp)




