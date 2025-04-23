
import torch

from PIL import Image
import torchvision.transforms as transforms

from torch.nn import functional as F

import os

from tqdm.auto import trange, tqdm
from einops import rearrange, repeat

import numpy as np
import pandas as pd

from transformers import CLIPProcessor, CLIPModel
from transformers import AutoImageProcessor, AutoModel
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import AutoProcessor, LlavaForConditionalGeneration
from random import randint

from torchvision import transforms

from einops import rearrange
import numpy as np
import random
from einops import rearrange, repeat
import json
import pdb
from bert_score import score
from openai import OpenAI
import httpx
import torch
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import base64
import re



class ZeroInsDataset(torch.utils.data.Dataset):
    def __init__(self, text_path, image_folder, dino_path):
        self.text_path = text_path
        self.image_folder = image_folder
        self.all_images = []
        for subdir, _, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.png')):
                    image_path = os.path.join(subdir, file)
                    self.all_images.append(image_path)
        #self.all_goals = [path.split(".txt")[0] for path in os.listdir("/mnt/data0/syc/step_data/sub_step_folder")]
        #self.all_goals = [path.split("_done.txt")[0] for path in os.listdir("/mnt/data4/syc/DiffusionGPT/sub_step_folder_done")]
        self.all_goals = [path.split(".txt")[0] for path in os.listdir(text_path)]
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_path)

        
    def __getitem__(self, index):
        cur_goal = self.all_goals[index].replace("_", " ") #os.path.dirname(cur_image).split("/")[-1].replace("_", " ")
        answer_file = os.path.join(self.text_path,self.all_goals[index] + '.txt') #+'_done.txt') #+'.txt')#
        cur_image_folder = os.path.join(self.image_folder, self.all_goals[index])
        with open(answer_file, 'r') as file:
            answer = file.read()

        try:
            relations = answer.split("*")[0].split(",")
            answer = answer.split("*")[1:-1]
            unrelated_0 = os.path.join(cur_image_folder,"Step_"+str(int(relations[0]))+".png")
            unrelated_1 = os.path.join(cur_image_folder,"Step_"+str(int(relations[0])+1)+".png")
            related_0 =  os.path.join(cur_image_folder,"Step_"+str(int(relations[1]))+".png")
            related_1 = os.path.join(cur_image_folder,"Step_"+str(int(relations[1])+1)+".png")
        except Exception as e:
            print(e)
            print("file:",answer_file)
        

        

        unrelated_0=self.dino_processor(images=Image.open(unrelated_0), return_tensors="pt")['pixel_values']
        unrelated_1=self.dino_processor(images=Image.open(unrelated_1), return_tensors="pt")['pixel_values']
        related_0=self.dino_processor(images=Image.open(related_0), return_tensors="pt")['pixel_values']
        related_1=self.dino_processor(images=Image.open(related_1), return_tensors="pt")['pixel_values']
        
        
        out = {'unrelated_0': unrelated_0, 
               'unrelated_1': unrelated_1, 
               'related_0': related_0, 
               'related_1': related_1, 
               'cur_goal':cur_goal}
        
        return out
    

    def __len__(self):
        return len(self.all_goals)


def custom_collate_fn(batch):
    #pdb.set_trace()
    unrelated_0 = torch.cat([item['unrelated_0'] for item in batch],dim=0)
    unrelated_1 = torch.cat([item['unrelated_1'] for item in batch],dim=0)
    related_0 = torch.cat([item['related_0'] for item in batch],dim=0)
    related_1 = torch.cat([item['related_1'] for item in batch],dim=0)
    cur_goal = [item['cur_goal'] for item in batch]  

    return {'unrelated_0': unrelated_0, 
            'unrelated_1': unrelated_1, 
            'related_0': related_0, 
            'related_1': related_1,  
            'cur_goal':cur_goal}


if __name__ == '__main__':
    # Shuffle the tensor
    torch.manual_seed(42)  # Set a seed for reproducibility
    random.seed(42)
    dataset = ZeroInsDataset(text_path="./step_data/all_tasks_with_logic",
                    image_folder="./results",
                    dino_path="./dinov2-large")
    
    dataset1 = ZeroInsDataset(text_path="./step_data/all_tasks_with_logic",
                    image_folder="./results",
                    dino_path="./dinov2-large")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=64,
        collate_fn=custom_collate_fn,
        num_workers=12
    )
    dataloader1 = torch.utils.data.DataLoader(
        dataset1,
        shuffle=False,
        batch_size=64,
        collate_fn=custom_collate_fn,
        num_workers=12
    )

    device="cuda:0"

    # dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    #dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    dino_processor = AutoImageProcessor.from_pretrained("./dinov2-large")
    dino_model = AutoModel.from_pretrained('./dinov2-large')
    # dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    dino_model = dino_model.to(device)
    dino_model.eval()


    torch.set_grad_enabled(False)


    # batched image based rating
    dino_score = []
    l2_score = []
    l2_r_score = []
    dino_score_1 = []
    l2_score_1 = []
    l2_r_score_1 = []
    all_low_list  = []
    for batch, batch1 in tqdm(zip(dataloader, dataloader1)): 
        
        torch.cuda.empty_cache()
        unrelated_0 = batch['unrelated_0'].cuda()  
        unrelated_1 = batch['unrelated_1'].cuda()  
        related_0 = batch['related_0'].cuda()  
        related_1 = batch['related_1'].cuda()  
        cur_goal = batch['cur_goal']

        unrelated_0_1 = batch1['unrelated_0'].cuda()  
        unrelated_1_1 = batch1['unrelated_1'].cuda()  
        related_0_1 = batch1['related_0'].cuda()  
        related_1_1 = batch1['related_1'].cuda()  
        cur_goal_1 = batch1['cur_goal']
        
        
        unrelated_0_features_1 = dino_model(pixel_values=unrelated_0_1).pooler_output
        unrelated_1_features_1 = dino_model(pixel_values=unrelated_1_1).pooler_output
        related_0_features_1 = dino_model(pixel_values=related_0_1).pooler_output
        related_1_features_1 = dino_model(pixel_values=related_1_1).pooler_output
        l2_r_1 =(unrelated_0_features_1 - unrelated_1_features_1)**2 +1e-8
        l2_1 =(related_0_features_1 - related_1_features_1)**2

        unrelated_0_features = dino_model(pixel_values=unrelated_0).pooler_output
        unrelated_1_features = dino_model(pixel_values=unrelated_1).pooler_output
        related_0_features = dino_model(pixel_values=related_0).pooler_output
        related_1_features = dino_model(pixel_values=related_1).pooler_output
        l2_r =(unrelated_0_features - unrelated_1_features)**2
        l2 =(related_0_features - related_1_features)**2
        
        # if l2_r_1.mean()< l2_r.mean() or l2_1.mean()>l2.mean():
        #    all_low_list.append(cur_goal)
        #    print("cur_goal: ",cur_goal, "cur_goal_1: ",cur_goal_1)
        #    print("l2: ",l2.mean(), "l2_1: ",l2_1.mean())
        #    print("l2_r: ",l2_r.mean(), "l2_r_1: ",l2_r_1.mean())
        #pdb.set_trace()

        # dino_score.append(cur_dino_score)
        # l2_score.append(l2.mean())
        # l2_r_score.append(l2_r.mean())
        # dino_score_1.append(cur_dino_score_1)
        # l2_score_1.append(l2_1.mean())
        # l2_r_score_1.append(l2_r_1.mean())
        l2_score.append(l2)
        l2_r_score.append(l2_r)
        l2_score_1.append(l2_1)
        l2_r_score_1.append(l2_r_1)
    
    print("l2 score:",torch.mean(torch.cat(l2_score).float()),"\n",)
    print("l2_r score:",torch.mean(torch.cat(l2_r_score).float()),"\n",)
    print("dino score:",torch.mean(torch.cat(l2_score).float())/torch.mean(torch.cat(l2_r_score).float()),"\n",)
    
    print("l2 score 1:",torch.mean(torch.cat(l2_score_1).float()),"\n",)
    print("l2_r score 1:",torch.mean(torch.cat(l2_r_score_1).float()),"\n",)
    print("dino score 1:",torch.mean(torch.cat(l2_score_1).float())/torch.mean(torch.cat(l2_r_score_1).float()),"\n",)
    #pdb.set_trace()
    # print(acc_metric.compute().item())