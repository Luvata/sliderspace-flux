import torch
from diffusers import FluxPipeline
import os
from tqdm import tqdm
import argparse


def ensure_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def gen_image(prompt, pipeline, seed, width, height, num_steps, batch_size):
    return pipeline(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=num_steps,
        max_sequence_length=256,
        height=height,
        width=width,
        generator=torch.Generator("cuda").manual_seed(seed),
        num_images_per_prompt=batch_size
    ).images

def generate_batch(prompt, pipeline, total_images, output_dir, width, height, num_steps, batch_size):
    ensure_directory(output_dir)
    
    num_batches = (total_images + batch_size - 1) // batch_size
    generated_count = 0
    
    for batch_idx in tqdm(range(num_batches)):
        seed = batch_idx
        images = gen_image(prompt, pipeline, seed, width, height, num_steps, batch_size)
        for i, image in enumerate(images):
            if generated_count >= total_images:
                break
            image.save(os.path.join(output_dir, f"image_seed{seed}_idx{i}.png"))
            generated_count += 1

def main():
    parser = argparse.ArgumentParser(description="Generate images using FluxPipeline.")
    parser.add_argument("prompt", type=str, help="The text prompt to generate images from.")
    parser.add_argument("-N", "--num_images", type=int, default=5000, help="Total number of images to generate.")
    parser.add_argument("--width", type=int, default=256, help="Width of generated images.")
    parser.add_argument("--height", type=int, default=256, help="Height of generated images.")
    parser.add_argument("--num_steps", type=int, default=4, help="Number of inference steps.")
    parser.add_argument("--bs", type=int, default=8, help="Batch size for image generation.")

    args = parser.parse_args()
    
    # Construct output directory e.g. images/a_picture_of_a_toy
    safe_prompt = "_".join(args.prompt.split())
    output_dir = os.path.join("images", safe_prompt)
    
    # Initialize pipeline
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
        device="cuda"
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    
    generate_batch(
        prompt=args.prompt,
        pipeline=pipe,
        total_images=args.num_images,
        output_dir=output_dir,
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        batch_size=args.bs
    )

if __name__ == "__main__":
    main()
