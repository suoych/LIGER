# coding: utf-8
import os
import random
import torch
import cv2
import re
import uuid
import json
import pickle
from PIL import Image, ImageDraw, ImageOps, ImageFont
import math
import numpy as np
import argparse
import inspect
import tempfile
import base64
# import subprocess

from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionInstructPix2PixPipeline
from diffusers import EulerAncestralDiscreteScheduler, PNDMScheduler, DPMSolverMultistepScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DiffusionPipeline, UniPCMultistepScheduler

from dragondiff.src.demo.edit import *

import ast
import shutil
from diffusers.models import AutoencoderKL
from transformers import CLIPProcessor, CLIPModel

from openai import OpenAI
import httpx

import gradio as gr
import requests
import sys
import pickle
from PIL import Image
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download
from datetime import datetime
from story_utils.gradio_utils import is_torch2_available
if is_torch2_available():
    from story_utils.gradio_utils import \
        AttnProcessor2_0 as AttnProcessor
else:
    from story_utils.gradio_utils  import AttnProcessor

import diffusers
from diffusers import StableDiffusionXLPipeline
from diffusers import DDIMScheduler
import torch.nn.functional as F
from story_utils.gradio_utils import cal_attn_mask_xl
import copy
from diffusers.utils import load_image
from story_utils.style_template import styles
from IPython.core.debugger import set_trace

from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from lisa_model.LISA import LISAForCausalLM
from lisa_model.llava import conversation as conversation_lib
from lisa_model.llava.mm_utils import tokenizer_image_token
from lisa_model.segment_anything.utils.transforms import ResizeLongestSide
from lisa_model.utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from story_inversion import *

#Each step must only include a single action of a single specific object. 


VERIFICATION_PROMPTS_FIRST_PART = """ We are generating illustrations for diffrent steps to make {input_overall}. 
The previous step is {input_pre}, and the image is:
"""


VERIFICATION_PROMPTS_CUR_PART = """
Our goal is to generate consisitent and illustrative instruction images. 
Now I wish you to evaluate the image quality according to the current step decription and previous step image based on salient subject object accuracy.
You need to identify does the salient subject object matches the description. We also expect if the steps has logic correlation and the object should look the same.

If error exists, you need to use different tools to fix it.
Tools include add, modify, remove.
Add is to add an object in the step description but not shown in the current image. Usage is Add(object description, place to add the object)
Modify is to make an object in the current image looks like the same as the object in the previous step image. Usage is Modify(object in current image, object in previous image)
Remove is to remove an redundant object from current image. Usage is Remove(object description)
Your response of the usage of the tools must start with *- and end with -*. For instance *-Remove(Apple)-*.

If the image is correct, you just answer Correct, no error. You must answer within 50 words and only correct the most obvious error with only one operation.

The current step is {input_cur}, and the image is:
"""


REGENERATE_PROMPT = """
Please evaluate the image quality according to the current step decription. You need to identify does the salient subject object matches the description.

If error exists, you need to use different tools to fix it.
Tools include add and remove.
Add is to add an object in the step description but not shown in the image. Usage is Add(object description, place to add the object)
Remove is to remove an redundant object from current image. Usage is Remove(object description)
Your response of the usage of the tools must start with *- and end with -*. For instance *-Remove(Apple)-*.

If the image is correct, you just answer Correct, no error. You must answer within 50 words and only correct the most obvious error with only one operation.

The current step is {input_cur}, and the image is:
"""


os.makedirs('image', exist_ok=True)

from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import AutoProcessor, AutoModelForVision2Seq

import numpy as np
from lama.lama_inpaint import inpaint_img_with_lama
from lama.utils import load_img_to_array, save_array_to_img, dilate_mask



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

#
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    
#################################################
########Consistent Self-Attention################
#################################################
class SpatialAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        text_context_len (`int`, defaults to 77):
            The context length of the text features.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self, hidden_size = None, cross_attention_dim=None,id_length = 4,device = "cuda:0",dtype = torch.float16):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1
        self.id_length = id_length
        self.id_bank = {}

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None):
        global total_count,attn_count,cur_step,mask1024,mask4096
        global sa32, sa64
        global write
        global height,width
        #print("hidden_shape:", hidden_states.shape)

        #set_trace()
        if write:
            # print(f"white:{cur_step}")
            self.id_bank[abs(cur_step)] = [hidden_states[:self.id_length], hidden_states[self.id_length:]] # abs for inversion
        else:
            encoder_hidden_states = torch.cat((self.id_bank[abs(cur_step)][0].to(self.device),hidden_states[:1],self.id_bank[abs(cur_step)][1].to(self.device),hidden_states[1:]))
            self.id_bank[abs(cur_step)] = [hidden_states[:self.id_length], hidden_states[self.id_length:]]
        # skip in early step
        if cur_step <-100: #change to 1?
            hidden_states = self.__call2__(attn, hidden_states,encoder_hidden_states,attention_mask,temb)
        else:   # 256 1024 4096
            random_number = random.random()
            # if cur_step <20:
            #     rand_num = 0.3
            # else:
            #     rand_num = 0.1
            # if random_number > rand_num:
            if not write:
                if hidden_states.shape[1] == (height//32) * (width//32):
                    attention_mask = mask1024[mask1024.shape[0] // self.total_length * self.id_length:]
                else:
                    attention_mask = mask4096[mask4096.shape[0] // self.total_length * self.id_length:]
            else:
                if hidden_states.shape[1] == (height//32) * (width//32):
                    attention_mask = mask1024[:mask1024.shape[0] // self.total_length * self.id_length,:mask1024.shape[0] // self.total_length * self.id_length]
                else:
                    attention_mask = mask4096[:mask4096.shape[0] // self.total_length * self.id_length,:mask4096.shape[0] // self.total_length * self.id_length]
            hidden_states = self.__call1__(attn, hidden_states,encoder_hidden_states,attention_mask,temb)
            # else:
            #     hidden_states = self.__call2__(attn, hidden_states,None,attention_mask,temb)
        attn_count +=1
        #print("cur_step: ", cur_step, "attn_count: ", attn_count)
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            mask1024,mask4096 = cal_attn_mask_xl(self.total_length,self.id_length,sa32,sa64,height,width, device=self.device, dtype= self.dtype)

        return hidden_states
    def __call1__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(total_batch_size, channel, height * width).transpose(1, 2)
        total_batch_size,nums_token,channel = hidden_states.shape
        img_nums = total_batch_size//2
        hidden_states = hidden_states.view(-1,img_nums,nums_token,channel).reshape(-1,img_nums * nums_token,channel)

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1,self.id_length+1,nums_token,channel).reshape(-1,(self.id_length+1) * nums_token,channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)


        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(total_batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)



        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)


        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(total_batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        # print(hidden_states.shape)
        return hidden_states
    def __call2__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, channel = (
            hidden_states.shape
        )
        # print(hidden_states.shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1,self.id_length+1,sequence_length,channel).reshape(-1,(self.id_length+1) * sequence_length,channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def set_attention_processor(unet,id_length):
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if name.startswith("up_blocks"):
                attn_procs[name] = SpatialAttnProcessor2_0(id_length = id_length)
            else:    
                attn_procs[name] = AttnProcessor()
        else:
            attn_procs[name] = AttnProcessor()

    unet.set_attn_processor(attn_procs)


def lisa_preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def get_reason_mask(image_path,text,tokenizer):
    conv = conversation_lib.conv_templates["llava_v1"].copy()
    conv.messages = []
    prompt = text
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    replace_token = (
        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    )
    prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    if not os.path.exists(image_path):
        print("File not found in {}".format(image_path))

    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    image_clip = (
        clip_image_processor.preprocess(image_np, return_tensors="pt")[
            "pixel_values"
        ][0]
        .unsqueeze(0)
        .cuda(device=0)
    )
    image_clip = image_clip.bfloat16()

    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]

    image = (
        lisa_preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        .unsqueeze(0)
        .cuda(device=0)
    )
    image = image.bfloat16()

    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda(device=0)

    #print(image_clip.device,image.device,input_ids.device,lisa_model.device)
    output_ids, pred_masks = lisa_model.evaluate(
        image_clip,
        image,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=512,
        tokenizer=tokenizer,
    )
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    #print("text_output: ", text_output)
    pred_mask = pred_masks[0].detach().cpu().numpy()[0]
    pred_mask = pred_mask > 0
    return pred_mask

def removing(image_path, mask):
    device = "cuda:0"
    img = load_img_to_array(image_path)
    mask = mask.astype(np.uint8) * 255
    mask = dilate_mask(mask, 15) 
    img_inpainted = inpaint_img_with_lama(img, mask, "./lama/configs/prediction/default.yaml", "./big-lama", device=device)
    return img_inpainted

def image_editing(text,input_files,output_file,edit_method,step,lisa_tokenizer,pipe=None):
    print("Editing the image:", input_files, ", based on the input text:", text, ", through the editing method:", edit_method)
    print("Saving the generated image to:", output_file)
    input_images = [Image.open(file) for file in input_files]
    
    if edit_method == "Modifying appearance":
        prompt = text
        n_prompt = ""
        mask_base = get_reason_mask(input_files[0],text[0],lisa_tokenizer)
        mask_replace = get_reason_mask(input_files[1],text[1],lisa_tokenizer)
        
        cv2.imwrite('mask_base.png', mask_base.astype(np.uint8) * 255)
        cv2.imwrite('mask_replace.png', mask_replace.astype(np.uint8) * 255)

        #print(mask_replace.dtype,mask_base.dtype)
        #print(type(input_images[0]),type(mask_base),type(input_images[1]),type(mask_replace))
        #result = appearance_modulation(image_base, point_base, image_replace, point_replace, prompt=" ", prompt_replace=" ")
        result = appearance_modulation(np.array(input_images[0]), mask_base.astype(np.uint8), np.array(input_images[1]), mask_replace.astype(np.uint8), prompt, prompt_replace=" ")
        result = Image.fromarray(result[0])
        result.save(output_file)
    elif edit_method == "Adding object":
        mask_base = get_reason_mask(input_files[0],text[1],lisa_tokenizer)
        result = pipe(
            prompt=text[0],
            image=input_images[0],
            mask_image=mask_base.astype(np.uint8)
        ).images[0]
        #img_filled = crop_for_filling_post(img, mask, np.array(img_crop_filled))
        result.save(output_file)
    elif edit_method == "Remove object":
        mask_remove = get_reason_mask(input_files[0],text[0],lisa_tokenizer)
        result = removing(input_files[0], mask_remove)
        result = Image.fromarray(result)
        #Image.fromarray(mask_remove).save("./mask.png")
        result.save(output_file)
    else:
        print("Unknown editing method.")

class Text2Instruction:
    def __init__(self):
        self.torch_dtype = torch.float16 
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.add_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "./stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        ).to('cuda:0')

    def llm(self, inputs):
        completion = self.client.chat.completions.create(
        model="gpt-4o",
        #model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "user", "content": inputs}
        ],
        #stream=True
        )
        #pdb.set_trace()
        output = completion.choices[0].message.content
        #output = [chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta.content is not None]
        #output = "".join(output)
        return output 

    def mllm(self, all_text,all_image):
        if len(all_image)<2:
            completion = self.client.chat.completions.create(
            #model="gpt-4o-mini",
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "user", "content": [
                    {"type": "text","text": all_text[0],},
                    {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{all_image[0]}","detail": "low"},},
                ]}
            ],)
            output = completion.choices[0].message.content
            return output 
        else:
            completion = self.client.chat.completions.create(
            #model="gpt-4o-mini",
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "user", "content": [
                    {"type": "text","text": all_text[0],},
                    {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{all_image[0]}","detail": "low"},},
                    {"type": "text","text": all_text[1],},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{all_image[1]}",  "detail": "low"},},
                ]}
            ],
            #stream=True
            )
            #pdb.set_trace()
            output = completion.choices[0].message.content
            #output = [chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta.content is not None]
            #output = "".join(output)
            return output 

    def mllm_verification(self, all_text,all_image_path):
        if len(all_image_path)<2:
            with open(all_image_path[0], "rb") as image_file:
                cur_image = base64.b64encode(image_file.read()).decode('utf-8')
            cur = REGENERATE_PROMPT.format(input_cur=all_text[0])
            output = self.mllm([cur], [cur_image])
            return output 
        else:
            with open(all_image_path[0], "rb") as image_file:
                previous_image_first = base64.b64encode(image_file.read()).decode('utf-8')
            with open(all_image_path[1], "rb") as image_file:
                cur_image = base64.b64encode(image_file.read()).decode('utf-8')
            image_list = [previous_image_first, cur_image]

            goal_pre = VERIFICATION_PROMPTS_FIRST_PART.format(input_overall=all_text[0],input_pre=all_text[1])
            cur = VERIFICATION_PROMPTS_CUR_PART.format(input_cur=all_text[2])
            text_list = [goal_pre, cur]
            output = self.mllm(text_list, image_list)
            #output = output.split("*")
            return output 
    
     


def extract_edit_method(text):
    # find all contents between #- and -*
    contents = []
    start = 0
    while start < len(text):
        start = text.find('*-', start)
        if start == -1:
            break  
        end = text.find('-*', start)
        if end == -1:
            break  
        contents.append(text[start + 1:end])
        start = end + 1
    return contents


def verificate_one_step(t2i, pipe, image_folder_path, question, cur_ans,lisa_tokenizer, pre_ans=None):
    print("Verificating: ",cur_ans.replace("\n", " "))
    if pre_ans is None:
        cur_step_name = cur_ans.split("Previous scene:")[0].strip().replace(" ","_").replace(":","")
        cur_image_name = os.path.join(image_folder_path, f"{cur_step_name}.png")
        verify_answer = t2i.mllm_verification([cur_ans.split("Action:")[1]], [cur_image_name])
        print("GPT judgement: ", verify_answer)
    else:
        pre_step_name = pre_ans.split("Previous scene:")[0].strip().replace(" ","_").replace(":","")
        cur_step_name = cur_ans.split("Previous scene:")[0].strip().replace(" ","_").replace(":","")
        cur_image_name = os.path.join(image_folder_path, f"{cur_step_name}.png")
        pre_image_name = os.path.join(image_folder_path, f"{pre_step_name}.png")
        verify_answer = t2i.mllm_verification([question, pre_ans.split("Action:")[1], cur_ans.split("Action:")[1]],
                                    [pre_image_name,cur_image_name])
        print("GPT judgement: ", verify_answer)
    
    all_action_list = extract_edit_method(verify_answer)
    if len(all_action_list)<1:
        return None
    else:
        action = all_action_list[0]
        print("Editing action: ", action)
        output_filename = os.path.join(image_folder_path, f"{cur_step_name}.png") # directly change
        if "Add" in action:
            step = cur_ans.split("Step")[-1].split(":")[0].strip()
            image_editing([action.split("(")[1].split(",")[0], action.split("(")[1].split(",")[1].split(")")[0]],
                [cur_image_name],output_filename,"Adding object",step,lisa_tokenizer,pipe=t2i.add_pipe)
        elif "Remove" in action:
            step = cur_ans.split("Step")[-1].split(":")[0].strip()
            image_editing([action.split("(")[1].split(")")[0]],
                [cur_image_name],output_filename,"Remove object",step,lisa_tokenizer)
        elif "Modify" in action:
            step = cur_ans.split("Step")[-1].split(":")[0].strip()
            image_editing([action.split("(")[1].split(",")[0], action.split("(")[1].split(",")[1].split(")")[0]],
                [cur_image_name, pre_image_name],output_filename,"Modifying appearance",step,lisa_tokenizer)
        else:
            print("Unknown action: ", action)
    
    if os.path.exists(output_filename):
        global cur_step, attn_count,write
        write = True
        cur_step = -49
        attn_count = 0
        inverted_latents = invert(pipe,prompt = cur_ans.split("Action:")[1]+cur_ans.split("Previous scene:")[1].split("\n")[0], 
                                  image = Image.open(output_filename).resize((768,768)),guidance_scale=1)
        inverted_latent = inverted_latents[0].clone().to(torch.float16).to("cuda:0")
        return inverted_latent
    else:
        return None
        
    


if __name__ == '__main__':

    global attn_count, total_count, id_length, total_length,cur_step, cur_model_type
    global write
    global  sa32, sa64
    global height,width
    attn_count = 0
    total_count = 0
    cur_step = 0
    id_length = 1
    total_length = 2
    cur_model_type = ""
    device="cuda:0"
    guidance_scale = 5
    num_steps = 50
    global attn_procs,unet
    attn_procs = {}
    write = False
    sa32 = 0.5
    sa64 = 0.5
    height = 768
    width = 768
    seed = 42
    global pipe
    global sd_model_path
    sd_model_path = "./sdxl-unstable-diffusers-y" 
    ### LOAD Stable Diffusion Pipeline
    #pipe = StableDiffusionXLPipeline.from_pretrained(sd_model_path, torch_dtype=torch.float16, use_safetensors=False)
    pipe = StableDiffusionXLPipeline.from_pretrained("./sdxl-unstable-diffusers-y", torch_dtype=torch.float16, use_safetensors=False)
    pipe = pipe.to(device)
    pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(50)
    unet = pipe.unet

    ### Insert PairedAttention
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None and (name.startswith("up_blocks") ) :
            attn_procs[name] =  SpatialAttnProcessor2_0(id_length = id_length)
            total_count +=1
        else:
            attn_procs[name] = AttnProcessor()
    print("successsfully load consistent self-attention")
    print(f"number of the processor : {total_count}")
    unet.set_attn_processor(copy.deepcopy(attn_procs))
    global mask1024,mask4096
    mask1024, mask4096 = cal_attn_mask_xl(total_length,id_length,sa32,sa64,height,width,device=device,dtype= torch.float16)

    # create lisa model
    lisa_tokenizer = AutoTokenizer.from_pretrained("./LISA-7B-v1-explanatory",cache_dir=None,model_max_length=512,padding_side="right",use_fast=False)
    lisa_tokenizer.pad_token = lisa_tokenizer.unk_token
    seg_token_idx = lisa_tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    torch_dtype = torch.bfloat16
    kwargs = {"torch_dtype": torch_dtype}
    lisa_model = LISAForCausalLM.from_pretrained("./LISA-7B-v1-explanatory", low_cpu_mem_usage=True, vision_tower="./clip-vit-large-patch14", seg_token_idx=seg_token_idx, **kwargs)
    lisa_model.config.eos_token_id = lisa_tokenizer.eos_token_id
    lisa_model.config.bos_token_id = lisa_tokenizer.bos_token_id
    lisa_model.config.pad_token_id = lisa_tokenizer.pad_token_id
    lisa_model.get_model().initialize_vision_modules(lisa_model.get_model().config)
    vision_tower = lisa_model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)
    lisa_model = lisa_model.bfloat16().cuda(device=0) # set to rank 1
    clip_image_processor = CLIPImageProcessor.from_pretrained(lisa_model.config.vision_tower)
    transform = ResizeLongestSide(1024)
    lisa_model.eval()


    seed_everything(42)
    t2i = Text2Instruction()
    setup_seed(seed)

    base_path = "./story_with_verification_results_sub"
    os.makedirs(base_path, exist_ok=True)
    for i,filename in enumerate(os.listdir('./step_data/all_tasks_with_logic')):
        print(i)
        if i>100:
            continue
        task = filename.split(".t")[0]
        question = task.replace("_"," ")

        task_folder = os.path.join(base_path,task)
        os.makedirs(task_folder, exist_ok=True)
        answer_file = os.path.join('./step_data/all_tasks_with_logic', filename)
        with open(answer_file, 'r') as file:
            answer = file.read()
        answer = answer.split("*")[:-1]
        torch.cuda.empty_cache()

        prompt_array = []
        step_name_array = []
        for ans in answer:
            ans = ans.strip("\n")
            step_name = ans.split("Previous scene:")[0].strip().replace(" ","_").replace(":","")
            try:
                if "None" in ans.split("Previous scene:")[1].split("\n")[0]:
                    subject = " "
                else:
                    subject = ans.split("Previous scene:")[1].split("\n")[0]
                action = ans.split("Action:")[1] #'top-down view, ' + ans.split("Action:")[1]
                prompt_array.append(action+subject) #subject + ", top-down view, " + action)
                step_name_array.append(step_name)
                
            except Exception as e:
                print(filename,"ans:", ans)
                print("An error occurred:", e)
                continue
        
        if len(prompt_array)< 1:
            print("error:", filename)
            continue

        negative_prompt = " "
        generator = torch.Generator(device="cuda").manual_seed(seed)
        id_prompts = prompt_array[:id_length]
        real_prompts = prompt_array[id_length:]
        torch.cuda.empty_cache()
        write = True
        cur_step = 0
        attn_count = 0
        print("id_prompts", id_prompts)
        id_images = pipe(id_prompts, num_inference_steps = num_steps, guidance_scale=guidance_scale, negative_prompt = negative_prompt, height = height, width = width,generator = generator).images
        id_images[0].save(os.path.join(task_folder, f"{step_name_array[0]}.png"))
        cur_ans = answer[0]
        try:
            latent = verificate_one_step(t2i, pipe, task_folder, question, cur_ans, lisa_tokenizer, None)
        except Exception as e:
            print(filename,"ans:", cur_ans)
            print("An error occurred:", e)
            
        for real_prompt,step_name in zip(real_prompts,step_name_array[1:]):
            write = False
            cur_ans = answer[int(step_name.split("_")[1])-1]
            pre_ans = answer[int(step_name.split("_")[1])-2]
            cur_step = 0
            print("real_prompt: ", real_prompt)
            real_image = pipe(real_prompt, num_inference_steps=num_steps, guidance_scale=guidance_scale,  height = height, width = width,negative_prompt = negative_prompt,generator = generator).images[0]
            real_image.save(os.path.join(task_folder, f"{step_name}.png")) 
            try:
                latent = verificate_one_step(t2i,pipe, task_folder, question,cur_ans, lisa_tokenizer, pre_ans)
            except Exception as e:
                print(filename,"ans:", cur_ans)
                print("An error occurred:", e)
        #pdb.set_trace()