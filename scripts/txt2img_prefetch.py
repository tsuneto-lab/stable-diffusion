from omegaconf import OmegaConf
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from ldm.util import instantiate_from_config

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
AutoFeatureExtractor.from_pretrained(safety_model_id)
StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
instantiate_from_config(config.model)

# Downloading: "https://github.com/DagnyT/hardnet/raw/master/pretrained/train_liberty_with_aug/checkpoint_liberty_with_aug.pth" to /root/.cache/torch/hub/checkpoints/checkpoint_liberty_with_aug.pth