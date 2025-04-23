
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

    def __init__(self, text_path, image_folder, clip_path, dino_path):
        self.text_path = text_path
        self.all_images = []
        for subdir, _, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.png')):
                    image_path = os.path.join(subdir, file)
                    self.all_images.append(image_path)
        self.all_goals = [path.split("_done.txt")[0] for path in os.listdir("./step_data/GT_description")]
        
        self.clip_processor = CLIPProcessor.from_pretrained(clip_path)
        self.tokenizer = self.clip_processor.tokenizer
        self.clip_image_processor = self.clip_processor.image_processor
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_path)

        
    def __getitem__(self, index):
        cur_image = self.all_images[index]

        cur_goal = os.path.dirname(cur_image).split("/")[-1].replace("_", " ")
        #answer_path = os.path.join(self.text_path, 'sub_step_folder')
        answer_path = self.text_path
        answer_file = os.path.join(answer_path,os.path.dirname(cur_image).split("/")[-1]+'_done.txt') #+'.txt')#
        
        with open(answer_file, 'r') as file:
            answer = file.read()
        answer = answer.split("*")[:-1]
        
        try:
            cur_answer_index = int(cur_image.split("_")[-1].split(".")[0])-1
            cur_step = answer[cur_answer_index].split("Action:")[0].split("Image Description:")[1]
            #cur_answer_index = int(cur_image.split("_")[-1].split(".")[0])-1
            #cur_step = answer[cur_answer_index].split("Action:")[1]
        except Exception as e:
            print(e)
            print("Error in ",cur_image)#, "cur_index: ", cur_answer_index)
            #print("answer: ",answer[cur_answer_index])
            print("answer length:", len(answer))
            #cur_step = answer[cur_answer_index-1]

        
        previous_step = int(cur_image.split("_")[-1].split(".")[0])-1
        previous_image = os.path.join(os.path.dirname(cur_image),"Step_"+str(previous_step)+".png")
        has_previous = True
        if not os.path.exists(previous_image):
            previous_image=self.dino_processor(images=Image.open(cur_image), return_tensors="pt")['pixel_values']
            has_previous=False
        else:
            previous_image=self.dino_processor(images=Image.open(previous_image), return_tensors="pt")['pixel_values']

        cur_image_clip = self.clip_processor(images=Image.open(cur_image), return_tensors="pt")['pixel_values']
        cur_image_kosmos = Image.open(cur_image) #self.kosmos_processor(images=Image.open(cur_image), return_tensors="pt")['pixel_values']
        #pdb.set_trace()
        cur_image_dino = self.dino_processor(images=Image.open(cur_image), return_tensors="pt")['pixel_values']
        
        out = {'cur_image_dino': cur_image_dino, 
               'cur_image_clip': cur_image_clip, 
               'cur_image_kosmos': cur_image_kosmos, 
               'cur_step':cur_step,
               'cur_goal':cur_goal,
               'has_previous':has_previous,
               'previous_image': previous_image,
               'cur_image_path': cur_image,}
        
        return out
    

    def __len__(self):
        return len(self.all_images)


def custom_collate_fn(batch):
    #pdb.set_trace()
    cur_image_dino = torch.cat([item['cur_image_dino'] for item in batch],dim=0)
    cur_image_clip = torch.cat([item['cur_image_clip'] for item in batch],dim=0)
    cur_image_kosmos = [item['cur_image_kosmos'] for item in batch] #torch.cat([item['cur_image_kosmos'] for item in batch],dim=0)
    cur_step = [item['cur_step'] for item in batch]  # list format
    cur_goal = [item['cur_goal'] for item in batch]  
    has_previous = [item['has_previous'] for item in batch] 
    previous_image = torch.cat([item['previous_image'] for item in batch],dim=0)
    cur_image_path = [item['cur_image_path'] for item in batch]  

    return {'cur_image_dino': cur_image_dino, 
            'cur_image_clip': cur_image_clip, 
            'cur_image_kosmos': cur_image_kosmos, 
            'has_previous':has_previous,
            'cur_step':cur_step,
            'cur_goal':cur_goal,
            'previous_image': previous_image,
            'cur_image_path': cur_image_path}


if __name__ == '__main__':
    # Shuffle the tensor
    torch.manual_seed(42)  # Set a seed for reproducibility
    random.seed(42)

    dataset = ZeroInsDataset(text_path="./step_data/GT_description",
                    image_folder="./step_data/results",
                    clip_path="./clip-vit-large-patch14",
                    dino_path="./dinov2-large")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=128,
        collate_fn=custom_collate_fn,
        num_workers=12
    )

    device="cuda:0"

    

    clip_model = CLIPModel.from_pretrained("./clip-vit-large-patch14")
    clip_processor = CLIPProcessor.from_pretrained("./clip-vit-large-patch14")
    clip_model = clip_model.to(device)
    clip_model.eval()

    blip2_processor = Blip2Processor.from_pretrained("./blip2-flan-t5-xl")
    blip2_model = Blip2ForConditionalGeneration.from_pretrained("./blip2-flan-t5-xl", torch_dtype=torch.float16, device_map="cuda")

    torch.set_grad_enabled(False)


    # batched image based rating
    #dino_score = []
    overall_goal_score = []
    all_clip_step_score = []
    all_clip_goal_score = []
    all_step_bert_P = []
    all_step_bert_R = []
    all_step_bert_F1 = []
   
    for batch_num, batch in enumerate(tqdm(dataloader)): 
        torch.cuda.empty_cache()
        cur_image_dino = batch['cur_image_dino'].cuda()  
        cur_image_clip = batch['cur_image_clip'].cuda()  
        cur_image_kosmos = batch['cur_image_kosmos']#.cuda()  
        has_previous = batch['has_previous']
        previous_image = batch['previous_image'].cuda()  
        cur_step = batch['cur_step']
        cur_goal = batch['cur_goal']
        cur_image_path = batch['cur_image_path']
        
        cur_step_id = clip_processor(text=cur_step, return_tensors="pt", padding=True, truncation=True)['input_ids'].cuda()
        cur_step_attn_mask = clip_processor(text=cur_step, return_tensors="pt", padding=True, truncation=True)['attention_mask'].cuda()
        cur_goal_id = clip_processor(text=cur_goal, return_tensors="pt", padding=True, truncation=True)['input_ids'].cuda()
        cur_goal_attn_mask = clip_processor(text=cur_goal, return_tensors="pt", padding=True, truncation=True)['attention_mask'].cuda()


        cur_step_similarity = clip_model(input_ids=cur_step_id, attention_mask=cur_step_attn_mask, pixel_values=cur_image_clip).logits_per_image
        #pdb.set_trace()
        clip_step_score =  torch.diag(cur_step_similarity)
        clip_step_score = torch.max(clip_step_score, torch.zeros_like(clip_step_score))
        all_clip_step_score.append(clip_step_score)

        
        inputs = blip2_processor(images=cur_image_kosmos, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)
        generated_ids = blip2_model.generate(**inputs)
        generated_text = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True) #[0].strip()
        P, R, F1 = score(generated_text, cur_step, lang='en', verbose=True)
        all_step_bert_P.append(P)
        all_step_bert_R.append(R)
        all_step_bert_F1.append(F1)
        

    print("clip step score:",torch.mean(torch.cat(all_clip_step_score).float()),"\n",)
    print("bert step score: ", "P:", torch.mean(torch.cat(all_step_bert_P).float()),
        "R:", torch.mean(torch.cat(all_step_bert_R).float()),
        "F1:", torch.mean(torch.cat(all_step_bert_F1).float()),"\n",)
    #pdb.set_trace()
    # print(acc_metric.compute().item())