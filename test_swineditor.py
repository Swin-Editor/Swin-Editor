import torch
import diffusers
import transformers
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import check_min_version
from diffusers import VQModel, AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from accelerate.logging import get_logger

import decord
decord.bridge.set_bridge('torch')
from einops import rearrange
from tqdm.auto import tqdm
import os
import argparse
import inspect
import math
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import logging

import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.utils import set_seed

from models.unet import SwinUnetModel
from util import save_videos_as_images, SwinPipe, inversion, save_videos



# Prepare data for training and inference
class SwinDataset(Dataset):
    """
    A custom dataset class for the Swin Transformer model.
    @param video_path - Path to the video file.
    @param prompt - The prompt for the video.
    @param frame_rate - Frame rate for sampling frames from the video.
    @param n_sample_frames - Number of frames to sample.
    @return A dictionary containing pixel values and prompt ids for the video.
    """
    def __init__(
            self,
            video_path: str,
            prompt: str,
            frame_rate: int = 1,
            n_sample_frames: int = 8,     
    ):
        self.width = self.height = 512
        self.frame_rate = frame_rate
        self.n_sample_frames = n_sample_frames
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = None


    def __len__(self):
        return 1

    def __getitem__(self, index):
        # load and sample video frames
        vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
        sample_index = list(range(0, len(vr), self.frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids
        }

        return example


logger = get_logger(__name__, log_level="INFO")
"""
    The main function that orchestrates the training process with various parameters and saves the trained model.
    @param train_data: Dictionary containing training data.
    @param validation_data: Dictionary containing validation data.
    @param output_dir: Directory to save the trained model.
    @param updated_modules: Tuple of strings specifying which modules to update during training.
    @param validation_steps: Number of steps before running validation. Default is 100.
    @param max_train_steps: Maximum number of training steps. Default is 500.
    @param max_grad_norm: Maximum gradient norm for gradient clipping. Default is 1.0.
    @param gradient_accumulation_steps: Number of steps to accumulate gradients before updating weights. Default is 1.
    @param checkpointing_steps: Number of steps before saving a checkpoint. Default is 
    """
def main(
    train_data: Dict,
    validation_data: Dict,
    output_dir: str,
    updated_modules: Tuple[str],
    validation_steps: int = 100,
    max_train_steps: int = 500,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,):
    
    # Fix the randomness to produce the same sequence of random numbers
    set_seed(33)
    *_, config = inspect.getargvalues(inspect.currentframe())
    

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="fp16",
    )
    
    # Initializing scheduler, tokenizer and models.
    pretrained_model_path = "checkpoints/stable-diffusion-v1-4"
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Register the output state for each samlpe
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))
    
    
    unet = SwinUnetModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")
    unet.requires_grad_(False)
    for name, module in unet.named_modules():
        if name.endswith(tuple(updated_modules)):
            for params in module.parameters():
                params.requires_grad = True

    unet.enable_xformers_memory_efficient_attention()
    unet.enable_gradient_checkpointing()

    # Initialize the optimizer
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(unet.parameters(), lr=3e-5, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-08,)

    # Get the training dataset
    train_dataset = SwinDataset(**train_data)

    # Preprocessing the dataset
    train_dataset.prompt_ids = tokenizer(
        train_dataset.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids[0]

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1
    )


    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)

    # Scheduler
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Managing our training everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    # Instantiate the validation pipeline
    validation_pipeline = SwinPipe(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    )
    validation_pipeline.enable_vae_slicing()

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    total_batch_size = 1 * accelerator.num_processes * gradient_accumulation_steps
    global_step = 0
    first_epoch = 0
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")
    
    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(output_dir, path))
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    

    for epoch in range(first_epoch, num_train_epochs):
        """
        Train a UNet model for a specified number of epochs using the provided data and settings.
        @param first_epoch - The starting epoch for training.
        @param num_train_epochs - The total number of epochs to train for.
        @param resume_from_checkpoint - Whether to resume training from a saved checkpoint.
        @param resume_step - The step to resume training from.
        @param gradient_accumulation_steps - Number of steps to accumulate gradients before updating the weights.
        @param train_dataloader - The dataloader for training data.
        @param accelerator - The accelerator for distributed training.
        @param weight_dtype - The data type for weights.
        @param vae - The Variational Autoencoder for encoding pixel values.
        @param noise_scheduler - The scheduler for adding noise to latents.
        @param text_encoder - The text encoder
        """
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert videos to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)
                video_length = pixel_values.shape[1]
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(1)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if global_step % validation_steps == 0:
                    if accelerator.is_main_process:
                        samples = []
                        generator = torch.Generator(device=latents.device)
                        generator.manual_seed(33)

                        ddim_inv_latent = None
                        if validation_data.use_inv_latent:
                            inv_latents_path = os.path.join(output_dir, f"inv_latents/ddim_latent-{global_step}.pt")
                            ddim_inv_latent = inversion(
                                validation_pipeline, ddim_inv_scheduler, video_latent=latents,
                                num_inv_steps=validation_data.num_inv_steps, prompt="")[-1].to(weight_dtype)
                            torch.save(ddim_inv_latent, inv_latents_path)

                        for idx, prompt in enumerate(validation_data.prompts):
                            sample = validation_pipeline(prompt, generator=generator, latents=ddim_inv_latent,
                                                         **validation_data).videos
                            save_videos(sample, f"{output_dir}/samples/sample-{global_step}/{prompt}.gif")
                            save_videos_as_images(sample, f"{output_dir}/samples/sample-{global_step}/{prompt}.gif",prompt)
                            samples.append(sample)
                        samples = torch.concat(samples)
                        save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                        save_videos(samples, save_path)
                        logger.info(f"Saved samples to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = SwinPipe.from_pretrained(
            pretrained_model_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(output_dir)

    accelerator.end_training()


"""
   The main function that loads a configuration file and runs the main function with the loaded configuration.
   If the script is run as the main program, it will parse the command-line arguments to get the configuration file path and then load and pass that configuration to the main function.
   @param config - The path to the configuration file
   @return None
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/swin.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))