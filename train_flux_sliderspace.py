#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np
import clip
from diffusers import FluxPipeline
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
import torch.nn.functional as F
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.loaders import FluxLoraLoaderMixin
from tqdm import tqdm

mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to("cuda")
std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to("cuda")


def latent_to_image(latents, pipe):
    latents = pipe._unpack_latents(latents, args.train_resolution, args.train_resolution, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return image

def clip_preprocess(image):
    image = F.interpolate(image, size=(224, 224), mode="bicubic", align_corners=False)
    image = ((image + 1.0) / 2.0).clamp(0.0, 1.0)
    image = (image - mean) / std
    return image


def train(args):
    clip_model, _ = clip.load("ViT-B/32", device="cuda")

    components = np.load(args.pca_path)
    os.makedirs(args.output_dir, exist_ok=True)

    component_ids = [int(x.strip()) for x in args.component_ids.split(",")]

    for comp_id in component_ids:
        print(f"\n=== Training on component {comp_id} ===")
        component = components[comp_id]
        component_vector = torch.tensor(component).to("cuda").unsqueeze(0)

        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt=args.prompt, prompt_2=None, max_sequence_length=256
            )

        del pipe.text_encoder
        del pipe.text_encoder_2
        torch.cuda.empty_cache()

        transformer, vae = pipe.transformer, pipe.vae
        transformer.requires_grad_(False)
        vae.requires_grad_(False)

        transformer_lora_config = LoraConfig(
            r=1, lora_alpha=1,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        transformer.add_adapter(transformer_lora_config)

        params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))
        n_params = sum(p.numel() for p in params_to_optimize)
        print(f"Component {comp_id}: Number of trainable parameters: {n_params}")

        optimizer = torch.optim.Adam(params_to_optimize, lr=args.learning_rate)
        pipe.text_encoder = pipe.transformer # to bypass the dtype check ¯\_(ツ)_/¯

        # Training loop.
        pbar = tqdm(range(1, args.num_train_steps + 1), desc=f"Component id={comp_id}")
        for step in pbar:
            transformer.disable_adapters()
            base_latents = pipe(
                guidance_scale=0,
                num_inference_steps=1,
                max_sequence_length=256,
                height=args.train_resolution,
                width=args.train_resolution,
                num_images_per_prompt=args.train_batchsize,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                output_type="latent",
                return_dict=False,
            )[0]

            base_image = clip_preprocess(latent_to_image(base_latents, pipe))
            base_embeds = clip_model.encode_image(base_image)

            optimizer.zero_grad()
            transformer.enable_adapters()
            # WARNING: Hacky bypass the @torch.no_grad() decorator in the pipeline.
            lora_latents = pipe.__call__.__wrapped__(
                pipe,
                guidance_scale=0,
                num_inference_steps=1,
                max_sequence_length=256,
                height=args.train_resolution,
                width=args.train_resolution,
                num_images_per_prompt=args.train_batchsize,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                output_type="latent",
                return_dict=False,
            )[0]

            lora_image = clip_preprocess(latent_to_image(lora_latents, pipe))
            lora_embeds = clip_model.encode_image(lora_image)

            direction = lora_embeds - base_embeds
            direction = direction / direction.norm(dim=-1, keepdim=True)

            loss = 1. - torch.nn.CosineSimilarity(eps=1e-6)(direction, component_vector).mean()
            loss.backward()

            grad_norm = torch.norm(torch.stack([p.grad.norm() for p in params_to_optimize]))
            optimizer.step()
            pbar.set_postfix({"loss": loss.item(), "grad_norm": grad_norm.item()})

            # Save a checkpoint every "save_every" steps.
            if step % args.save_every == 0:
                lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(transformer))
                save_dir = os.path.join(args.output_dir, f"component_{comp_id}_step_{step}")
                os.makedirs(save_dir, exist_ok=True)
                FluxLoraLoaderMixin.save_lora_weights(save_dir, transformer_lora_layers=lora_layers_to_save)

        # Cleanup for this component.
        del pipe, lora_image, lora_embeds, base_latents, lora_latents, pooled_prompt_embeds, transformer, vae
        print(f"Finished training component {comp_id} at step {step}.")
        print("Max memory allocated (GB):", torch.cuda.max_memory_allocated() / 1e9)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LoRA adapters for a FluxPipeline using specific PCA component IDs."
    )
    parser.add_argument("--pca_path", type=str, required=True,
                        help="Path to the PCA .npy file containing precomputed PCA components.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt text to use for training.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save trained checkpoints.")
    parser.add_argument("--num_train_steps", type=int, default=100,
                        help="Number of training steps per component (default: 100).")
    parser.add_argument("--train_resolution", type=int, default=256,
                        help="Resolution for training images (default: 256).")
    parser.add_argument("--train_batchsize", type=int, default=4,
                        help="Training batch size (default: 4).")
    parser.add_argument("--component_ids", type=str, required=True,
                        help="Comma-separated list of PCA component IDs to train (e.g., '0,1,2,3,6,7,15,21').")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Frequency (in steps) at which to save checkpoints (default: 100).")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for the optimizer (default: 1e-4).")
    args = parser.parse_args()
    train(args)