import numpy as np
from dragondiff.src.demo.utils import segment_only_with_points, segment_only_with_points_paste
from dragondiff.src.demo.model import DragonModels
import pdb
from PIL import Image
import cv2


pretrained_model_path = "./StableDiffusion/"
model = DragonModels(pretrained_model_path=pretrained_model_path)

def move_object(original_image, global_points,
                prompt=" ", resize_scale=1, w_edit=4, 
                w_content=6, w_contrast=0.2, w_inpaint=0.8, 
                seed=42, selected_points=[(0,0),(256,256)], 
                guidance_scale=4, energy_scale=0.5, 
                max_resolution=512, SDE_strength=0.4, ip_scale=0.1):
    """
    ## Object Moving & Resizing
    Usage:
    - Upload a source image, and then draw a box to generate the mask corresponding to the editing object.
    - Label the object's movement path on the source image.
    - Label reference region. (optional)
    - Add a text description to the image and click the `Edit` button to start editing.
    """
    #pdb.set_trace()
    original_image = cv2.imread(original_image)
    original_image, mask = segment_only_with_points(original_image, global_points)
    result = model.run_move(original_image, mask, mask_ref=None,
                   prompt=prompt, resize_scale=resize_scale, w_edit=w_edit, 
                   w_content=w_content, w_contrast=w_contrast, w_inpaint=w_inpaint, 
                   seed=seed, selected_points=selected_points, 
                   guidance_scale=guidance_scale, energy_scale=energy_scale, 
                   max_resolution=max_resolution, SDE_strength=SDE_strength, ip_scale=ip_scale)
    #pdb.set_trace()
    return result

def appearance_backup(image_base, point_base, image_replace, point_replace, 
                          prompt=" ", prompt_replace=" ", w_edit=3.5, w_content=5, 
                          seed=42, guidance_scale=5, energy_scale=2, max_resolution=512, 
                          SDE_strength=0.4, ip_scale=0.1):
    DESCRIPTION = """
    ## Appearance Modulation
    Usage:
    - Upload a source image, and an appearance reference image.
    - Label object masks on these two image.
    - Add a text description to the image and click the `Edit` button to start editing."""
    
    image_base = np.array(Image.open(image_base))
    image_replace = np.array(Image.open(image_replace))
    image_base, mask_base = segment_only_with_points(image_base, point_base)
    image_replace, mask_replace = segment_only_with_points(image_replace, point_replace)
    result = model.run_appearance(image_base, mask_base, image_replace, mask_replace, 
                                  prompt, prompt_replace, w_edit, w_content, seed, 
                                  guidance_scale, energy_scale, max_resolution, 
                                  SDE_strength, ip_scale)
    #pdb.set_trace()
    return result

def appearance_modulation(image_base, mask_base, image_replace, mask_replace, 
                          prompt=" ", prompt_replace=" ", w_edit=3.5, w_content=5, 
                          seed=42, guidance_scale=5, energy_scale=2, max_resolution=512, 
                          SDE_strength=0.4, ip_scale=0.1):
    DESCRIPTION = """
    ## Appearance Modulation
    Usage:
    - Upload a source image, and an appearance reference image.
    - Label object masks on these two image.
    - Add a text description to the image and click the `Edit` button to start editing."""
    
    #image_base = np.array(Image.open(image_base))
    #image_replace = np.array(Image.open(image_replace))
    result = model.run_appearance(image_base, mask_base, image_replace, mask_replace, 
                                  prompt, prompt_replace, w_edit, w_content, seed, 
                                  guidance_scale, energy_scale, max_resolution, 
                                  SDE_strength, ip_scale)
    #pdb.set_trace()
    return result

def object_pasting(image_replace, point_replace, image_base, 
                   prompt, prompt_replace, w_edit=4, w_content=6, 
                   seed=42, guidance_scale=4, energy_scale=1.5, dx=0, dy=0, 
                   resize_scale=1, max_resolution=512, SDE_strength=0.4, ip_scale=0.1):
    DESCRIPTION = """
    ## Object Pasting
    Usage:
    - Upload a reference image, having the target object.
    - Label object masks on the reference image.
    - Upload a background image.
    - Modulate the size and position of the object after pasting.
    - Add a text description to the image and click the `Edit` button to start editing."""
    #dx and dy are horizontal and vertical movement, ranges from -1000 to 1000
    
    image_base = np.array(Image.open(image_base))
    image_replace = np.array(Image.open(image_replace))
    mask_base_show, mask_replace = segment_only_with_points_paste(image_replace, point_replace, image_base, dx, dy, resize_scale)
        
    result = model.run_paste(image_base, mask_replace, image_replace, 
                             prompt, prompt_replace, w_edit, w_content, 
                             seed, guidance_scale, energy_scale, dx, dy, 
                             resize_scale, max_resolution, SDE_strength, ip_scale)
    #pdb.set_trace()
    return result