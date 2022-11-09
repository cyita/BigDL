from addict import Dict
import inspect

from diffusers import StableDiffusionPipeline
import torch

from examples.unet_accuracy import Accuracy
from unet_engine import UNETEngine
from openvino.runtime import Core
from openvino.tools.pot.graph import load_model, save_model
from openvino.tools.pot.engines.ie_engine import IEEngine
from openvino.tools.pot.pipeline.initializer import create_pipeline
from openvino.tools.pot.graph.model_utils import compress_model_weights
from openvino.tools.pot.utils.logger import init_logger, get_logger
from bigdl.nano.pytorch import InferenceOptimizer
from torch.utils.data import TensorDataset, DataLoader, Dataset


pipe = StableDiffusionPipeline.from_pretrained("/home/yina/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-4/snapshots/f15bc7606314c6fa957b4267bee417ee866c0b84")
prompt = "a photo of an astronaut riding a horse on mars"
# output = pipe(prompt, guidance_scale=7.5)
# # samples_for_unet = output[1]
# image = output[0]
# image[0][0].save("astronaut_rides_horse_ipex_original.png")
# print(len(samples_for_unet))
# print(pipe.device)
# print(pipe.unet.device)
prompts = ["a photo of an astronaut riding a horse on mars"]
samples_for_unet = []
for p in prompts:
    samples = pipe(prompt, guidance_scale=1, return_latents=True)[1]
    samples_for_unet.append(samples[0])
print(len(samples_for_unet))

# input_names = ["sample", "timestep", "encoder_hidden_states"]

class  MyDataset(Dataset):
    def __init__(self, sample):
        self.sample = sample
    
    def __len__(self):
        return len(self.sample)

    def __getitem__(self, index):
        data = self.sample[index]
        return {"sample": data[0], "timestep": data[1], "encoder_hidden_states": data[2]}, None

dataset = MyDataset(samples)

def func(data):
    return data[0]

loader = DataLoader(dataset, batch_size=1, collate_fn=func)

sample_latents = torch.randn((1, 4, 64, 64), generator=None, device="cpu", dtype=torch.float32)

algorithms = [
	{
		"name": "DefaultQuantization",
		"params": {
			"target_device": "CPU",
			"preset": "accuracy",
			# "stat_subset_size": 300,
			"stat_batch_size": 1,
			"model_type": "transformer",
		},
	}
]

ov_quan = InferenceOptimizer.quantize(model=pipe.unet, accelerator="openvino", calib_dataloader=loader, 
                            input_sample=(torch.cat([sample_latents]), 980, torch.randn((1, 77, 768), generator=None, device="cpu", dtype=torch.float32), True),
                            input_names=["sample", "timestep",
                            "encoder_hidden_states", "return_dict"],
                             output_names=["unet_output"],
                             dynamic_axes={"sample": [0],
                             "encoder_hidden_states": [0],
                             "unet_output": [0]})
