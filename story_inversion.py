from torchvision import transforms as tfms
import torch
import os
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    rescale_noise_cfg,
    StableDiffusionXLPipelineOutput,
)
import diffusers
from diffusers import StableDiffusionXLPipeline
from diffusers import DDIMScheduler
import torch.nn.functional as F
from tqdm.auto import tqdm

def _backward_ddim(x_tm1, alpha_t, alpha_tm1, eps_xt):
    """
    let a = alpha_t, b = alpha_{t - 1}
    We have a > b,
    x_{t} - x_{t - 1} = sqrt(a) ((sqrt(1/b) - sqrt(1/a)) * x_{t-1} + (sqrt(1/a - 1) - sqrt(1/b - 1)) * eps_{t-1})
    From https://arxiv.org/pdf/2105.05233.pdf, section F.
    """
    a, b = alpha_t, alpha_tm1
    sa = a**0.5
    sb = b**0.5
    return sa * ((1 / sb) * x_tm1 + ((1 / a - 1) ** 0.5 - (1 / b - 1) ** 0.5) * eps_xt)


@torch.no_grad()
def invert(
    pipe,
    prompt= None,
    prompt_2= None,
    image= None,
    guidance_scale=1,
    num_inference_steps= 50,
    negative_prompt = None,
    negative_prompt_2 = None,
    num_images_per_prompt = 1,
    latents = None,
    prompt_embeds = None,
    negative_prompt_embeds = None,
    pooled_prompt_embeds = None,
    negative_pooled_prompt_embeds = None,
    cross_attention_kwargs = None,
    original_size = None,
    crops_coords_top_left = (0, 0),
    target_size = None,
    negative_original_size = None,
    negative_crops_coords_top_left= (0, 0),
    negative_target_size = None,
):
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipe._execution_device

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = True
    # 3. Encode input prompt
    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None)
        if cross_attention_kwargs is not None
        else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
    )

    # 4. Preprocess image
    image = pipe.image_processor.preprocess(image)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    
    # 6. Prepare latent variables
    pipe.vae.to(torch.float32)
    with torch.no_grad():
        latents = pipe.vae.encode(image.to(torch.float32).to(device))
    pipe.vae.to(torch.float16)
    print(latents.latent_dist.sample().shape, latents.latent_dist.sample().min(),latents.latent_dist.sample().max(), pipe.scheduler.init_noise_sigma)
    latents = 0.13025 * latents.latent_dist.sample() #latents.latent_dist.sample() * self.scheduler.init_noise_sigma
    latents = latents.to(torch.float16)
    
    
    height, width = latents.shape[-2:]
    height = height * pipe.vae_scale_factor
    width = width * pipe.vae_scale_factor

    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    # 8. Prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds

    if pipe.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim
    print(text_encoder_projection_dim)
    add_time_ids = pipe._get_add_time_ids(original_size, crops_coords_top_left, target_size, prompt_embeds.dtype,text_encoder_projection_dim)
    add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device)
    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = pipe._get_add_time_ids(
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    prev_timestep = None

    for t in tqdm(reversed(pipe.scheduler.timesteps)):
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


        alpha_prod_t = pipe.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (
            pipe.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep is not None
            else pipe.scheduler.final_alpha_cumprod
        )
        prev_timestep = t
        #print(latents.shape, latents.min(),latents.max())
        latents = _backward_ddim(
            x_tm1=latents,
            alpha_t=alpha_prod_t,
            alpha_tm1=alpha_prod_t_prev,
            eps_xt=noise_pred,
        )

    image = latents
    return StableDiffusionXLPipelineOutput(images=image)