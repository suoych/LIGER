
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
import multiprocessing

RATING_PROMPT = """ Rate the image from 1 (worst) to 5 (perfect) considering: 
A. Does the image contains the objects should appear for the text description?
B. The image does not contain unrelated objects?
C. According to the text description, imagine the subject object attribute (adjective, state, color, texture), and does the image show correct attributes? 
Give a rate from 1 to 5 on each aspect within 30 words in a format like A:rating*.
The text description is {input_overall} and the image is:
"""


QUALITY_PROMPT = """ Rate the image from 1 (worst) to 5 (perfect) considering the image quality and detail.
Give a rate number from 1 to 5 and a reason within 30 words.
"""
COHERENT_STATE_CHANGE_PROMPT = """ Please rate the series images from 1 (worst) to 5 (ok) considering:
A. In some consecutive steps, the image are coherent.
B. The image is diverse when the text descriptions deviates.
C. Overall, can the whole image series roughly describe the coarse idea of the task?
Give a rate from 1 to 5 on each aspect. Do not be too strict since the task is hard. The response should be in a format of A:(number of rating)* Reason: (reasons).
Considering the task of {input_task}. The text description for each step is {input_overall} and the image series are:
"""

I2T_PROMPT = """Describe the image in detail within 50 words:"""

SUMMARY_PROMPT = """According to the following descriptions of a series of images, summarize an overall description with in 150 words. {input_overall} """


def single_image_evaluation(shared_list1, shared_list2, shared_list3,shared_list4, shared_list5, image_path):

    try:
        with open(image_path, "rb") as image_file:
            cur_image = base64.b64encode(image_file.read()).decode('utf-8')
        #text_verification_part = RATING_PROMPT.format(input_overall=cur_step)
        content_list = [{"type": "text","text": QUALITY_PROMPT},
            {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{cur_image}","detail": "low"},},]
        
        completion = client.chat.completions.create(model="gpt-4o", temperature=1,
        messages=[
            {"role": "user", "content": content_list}
        ],)
        output = completion.choices[0].message.content
        #print("Image:", image_path, "\n", "Output:", output)
        numbers = re.findall(r'\d+', output)
        numbers = [int(i) for i in numbers]
        print(image_path, output)
        if len(numbers) <1 or min(numbers) < 1 or max(numbers)>5:
            print("Error in: ", image_path)
            numbers = [3]
        #else:
        shared_list1.append(numbers[0])
            #shared_list2.append(numbers[1])
            #shared_list3.append(numbers[2])
        shared_list2.append(image_path)
        shared_list3.append(output)
    except Exception as e:
        print("Error in: ", image_path)
        shared_list1.append(3)
            #shared_list2.append(numbers[1])
            #shared_list3.append(numbers[2])
        shared_list2.append(image_path)
        shared_list3.append("error!")

def multi_image_evaluation(shared_list1, shared_list2, shared_list3,shared_list4, shared_list5,text_path,image_base):

    answer_path = "step_data/all_tasks_with_logic" 
    answer_file = os.path.join(answer_path,text_path)
    with open(answer_file, 'r') as file:
        answer = file.read()
    answer = answer.split("*")[:-1]
    all_steps = ""
    for item in answer:
        all_steps = all_steps + str(item.split("Action:")[1])+"\n" #str(item.split("Action:")[0].split("Image Description:")[1]) + "\n"

    image_folder = os.path.join(image_base, text_path.split("_done.txt")[0])
    image_list = [os.path.join(image_folder,image_name) for image_name in os.listdir(image_folder)]
    all_image_64 = []
    for image_path in image_list:
        with open(image_path, "rb") as image_file:
            cur_image = base64.b64encode(image_file.read()).decode('utf-8')
            all_image_64.append(cur_image)
    text_verification_part = COHERENT_STATE_CHANGE_PROMPT.format(input_task=text_path.split("_done.txt")[0].replace("_"," "),input_overall=all_steps)
    content_list = [{"type": "text","text": text_verification_part},]
    for cur_image in all_image_64:
        content_list.append({"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{cur_image}","detail": "low"},})
    
    completion = client.chat.completions.create(model="gpt-4o", temperature=1,
    messages=[
        {"role": "user", "content": content_list}
    ],)
    output = completion.choices[0].message.content
    #print("Task:", text_path, "\n", "Output:", output)
    numbers = re.findall(r'\d+', output)
    numbers = [int(i) for i in numbers]
    if len(numbers) != 3 or min(numbers) < 1 or max(numbers)>5:
        print("Error in: ", image_path)
    else:
        shared_list1.append(numbers[0])
        shared_list2.append(numbers[1])
        shared_list3.append(numbers[2])
        shared_list4.append(text_path)
        shared_list5.append(output)

def i2t_evaluation(shared_list1, shared_list2, shared_list3, shared_list4, shared_list5, shared_list6, text_path,image_base):
    cur_goal = text_path.split("_done.txt")[0].replace("_", " ")
    answer_path = "step_data/all_tasks_with_logic" 
    answer_file = os.path.join(answer_path,text_path)
    with open(answer_file, 'r') as file:
        answer = file.read()
    answer = answer.split("*")[:-1]
    all_steps_list = [item.split("Action:")[0].split("Image Description:")[1] for item in answer]
    all_steps = ""
    for item in answer:
        all_steps = all_steps + str(item.split("Action:")[0].split("Image Description:")[1]) + "\n"

    image_folder = os.path.join(image_base, text_path.split("_done.txt")[0])
    image_list = [os.path.join(image_folder,image_name) for image_name in os.listdir(image_folder)]
    all_captions = []
    for image_path in image_list:
        with open(image_path, "rb") as image_file:
            cur_image = base64.b64encode(image_file.read()).decode('utf-8')
        content_list = [{"type": "text","text": I2T_PROMPT},
                        {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{cur_image}","detail": "low"},}]
    
        completion = client.chat.completions.create(model="gpt-4o", temperature=1,
        messages=[
            {"role": "user", "content": content_list}
        ],)
        output = completion.choices[0].message.content
        all_captions.append(output)
    #summary
    all_caption_str = ""
    for i in all_captions:
        all_caption_str = all_caption_str + i
    text_verification_part = RATING_PROMPT.format(input_overall=all_caption_str)
    content_list = [{"type": "text","text": text_verification_part},]
    completion = client.chat.completions.create(model="gpt-4o", temperature=1,
    messages=[
        {"role": "user", "content": content_list}
    ],)
    summary = completion.choices[0].message.content
    print("Task:", text_path, "\n", "Output:", output)
    
    P, R, F1 = score(all_captions, all_steps_list, lang='en', verbose=True)
    P_G, R_G, F1_G = score([summary], [cur_goal], lang='en', verbose=True)
    shared_list1.append(P)
    shared_list2.append(R)
    shared_list3.append(F1)
    shared_list4.append(P_G)
    shared_list5.append(R_G)
    shared_list6.append(F1_G)



if __name__ == '__main__':
    # Shuffle the tensor
    torch.manual_seed(42)  # Set a seed for reproducibility
    random.seed(42)

    
    image_folder = "./results"
    text_folder="./step_data/all_tasks_with_logic"

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    device="cuda:0"
    torch.set_grad_enabled(False)

    mode = "single"

    if mode == "single":
        all_images = []
        for subdir, _, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.png')):
                    image_path = os.path.join(subdir, file)
                    all_images.append(image_path)

        with multiprocessing.Manager() as manager:
            shared_list1 = manager.list()
            shared_list2 = manager.list()
            shared_list3 = manager.list()
            shared_list4 = manager.list()
            shared_list5 = manager.list()

            with multiprocessing.Pool(processes=500) as pool:
                jobs = []
                for item in all_images:
                    job = pool.apply_async(single_image_evaluation, (shared_list1, shared_list2, shared_list3, shared_list4, shared_list5, item))
                    jobs.append(job)

                for job in tqdm(jobs):
                    job.get()  

            A, all_path,all_response = list(shared_list1), list(shared_list2), list(shared_list3)#, list(shared_list4), list(shared_list5)
        print("all_path:", all_path)
        print("all_response:", all_response)
        print("A: ", sum(A)/len(A), " In ", len(A),"items.")
        #print("B: ", sum(B)/len(B), " In ", len(B),"items.")
        #print("C: ", sum(C)/len(C), " In ", len(C),"items.")
        pdb.set_trace()
    elif mode == "series":
        all_text = os.listdir(text_folder)
        with multiprocessing.Manager() as manager:
            shared_list1 = manager.list()
            shared_list2 = manager.list()
            shared_list3 = manager.list()
            shared_list4 = manager.list()
            shared_list5 = manager.list()

            with multiprocessing.Pool(processes=50) as pool:
                jobs = []
                for item in all_text:
                    job = pool.apply_async(multi_image_evaluation, (shared_list1, shared_list2, shared_list3, shared_list4, shared_list5, item, image_folder))
                    jobs.append(job)

                for job in tqdm(jobs):
                    job.get()  

            A, B, C, all_path,all_response = list(shared_list1), list(shared_list2), list(shared_list3), list(shared_list4), list(shared_list5)
            print("all_path:", all_path)
            print("all_response:", all_response)
            print("A: ", sum(A)/len(A), " In ", len(A),"items.")
            print("B: ", sum(B)/len(B), " In ", len(B),"items.")
            print("C: ", sum(C)/len(C), " In ", len(C),"items.")
        pdb.set_trace()
    elif mode == "I2T":
        all_text = os.listdir("./step_data/all_tasks_with_logic")
        
        with multiprocessing.Manager() as manager:
            shared_list1 = manager.list()
            shared_list2 = manager.list()
            shared_list3 = manager.list()
            shared_list4 = manager.list()
            shared_list5 = manager.list()
            shared_list6 = manager.list()

            with multiprocessing.Pool(processes=20) as pool:
                jobs = []
                for item in all_text:
                    job = pool.apply_async(i2t_evaluation, (shared_list1, shared_list2,shared_list3, shared_list4,shared_list5, shared_list6, item, image_folder))
                    jobs.append(job)

                for job in tqdm(jobs):
                    job.get()  

            P, R, F1, P_G, R_G, F1_G = list(shared_list1), list(shared_list2), list(shared_list3), list(shared_list4), list(shared_list5), list(shared_list6)
            print("P: ", torch.cat(P).mean(), " In ", torch.cat(P).shape,"items.")
            print("R: ", torch.cat(R).mean(), " In ", torch.cat(R).shape,"items.")
            print("F1: ", torch.cat(F1).mean(), " In ", torch.cat(F1).shape,"items.")
            print("P_G: ", torch.cat(P_G).mean(), " In ", torch.cat(P_G).shape,"items.")
            print("R_G: ", torch.cat(R_G).mean(), " In ", torch.cat(R_G).shape,"items.")
            print("F1_G: ", torch.cat(F1_G).mean(), " In ", torch.cat(F1_G).shape,"items.")
        pdb.set_trace()
    else:
        print("Error!")
        pdb.set_trace()
