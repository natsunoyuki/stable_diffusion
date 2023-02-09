import os
import diffusers
import torch
import numpy as np
import matplotlib.pyplot as plt


# Load (text to image) diffuser pipeline.
def load_pipeline(model_dir, scheduler = None, device_name = torch.device("cpu")):
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(model_dir, torch_dtype = torch.float32)

    if scheduler is None or scheduler in ["EulerAncestralDiscreteScheduler", "EADS"]:
        pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler in ["EulerDiscreteScheduler", "EDS"]:
        pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler in ["DPMSolverMultistepScheduler", "DPMSMS"]:
        pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.safety_checker = lambda images, **kwargs: [images, [False] * len(images)]
    pipe = pipe.to(device_name)
    return pipe


# Load image to image diffuser pipeline.
def load_img2img_pipeline(model_dir, device_name):
    pipe = diffusers.StableDiffusionImg2ImgPipeline.from_pretrained(model_dir, torch_dtype = torch.float16)
    pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = lambda images, **kwargs: [images, [False] * len(images)]
    pipe = pipe.to(device_name)
    return pipe


# Run diffuser pipeline.
def run_pipe(pipe, prompt, negative_prompt = None, steps = 60, 
             width = 512, height = 704, scale = 8.0, seed = 123, n_images = 1,
             device_name = torch.device("cpu")):
    if width % 8 != 0:
        print("Image width must be multiples of 8... adjusting!")
        width = int(width / 8) * 8
    if height % 8 != 0:
        print("Image width must be multiples of 8... adjusting!")
        height = int(height / 8) * 8

    if device_name == torch.device("mps"):
        gen = torch.Generator().manual_seed(seed)
    else:
        gen = torch.Generator(device = device_name).manual_seed(seed)
    image_list = []

    with torch.autocast("cuda"): 
        for i in range(n_images): 
            image = pipe(prompt, height = height, width = width, 
                         num_inference_steps = steps, guidance_scale = scale,
                         negative_prompt = negative_prompt, generator = gen)
            image_list = image_list + image.images

    return image_list


# Plot pipeline outputs.
def plot_images(images, labels = None):
    N = len(images)
    n_cols = 5
    n_rows = int(np.ceil(N / n_cols))

    plt.figure(figsize = (20, 5 * n_rows))
    for i in range(len(images)):
        plt.subplot(n_rows, n_cols, i + 1)
        if labels is not None:
            plt.title(labels[i])
        plt.imshow(np.array(images[i]))
        plt.axis(False)
    plt.show()


def save_images(image_list, image_names = None, save_dir = "./"):
    if image_names is None:
        image_names = ["{}.png".format(i) for i in range(len(image_list))]

    assert len(image_list) == len(image_names)

    for i in range(len(image_list)):
        image_list[i].save(os.path.join(save_dir, image_names[i]))