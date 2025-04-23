# LIGER

Code for the paper Long-horizon Visual Instruction Generation with Logic and Attribute Self-reflection (ICLR 2025)


## Preparation
Prepare the following projects according to their instructions:

* [Lama](https://github.com/advimman/lama)

* [LISA](https://github.com/dvlab-research/LISA)

* [SD-Inpaint](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)

* [DragonDiffusion](https://github.com/MC-E/DragonDiffusion)

Then install the requirements by running:

```
pip install -r requirements.txt
```

Prepare your Openai API key by:

```
export OPENAI_API_KEY='Your_API_Key'
```


## Proposed Dataset

The proposed dataset contains 569 long-horizon tasks are in the ```step_data``` folder. 

The ```GT_description``` subfolder contains the annotated descriptions for each step in every task.

The ```all_tasks_with_logic``` subfolder contains the GPT generated task step descriptions, historical prompt, and logic step annotations.

## Running

Run following command for instruction generation:

```
python run.py
```

## Evaluation

Run following command for evaluation:

```
python metric_clip_and_text.py    #CLIP-score and BERT-score evaluation
python metric_dino.py             #DINO-score evaluation
python metric_GPT.py              #GPT evaluation
```
