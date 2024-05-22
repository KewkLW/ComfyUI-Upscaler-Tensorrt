import os
import folder_paths
import numpy as np
import torch
from comfy.utils import ProgressBar
from .utilities import Engine

ENGINE_DIR = os.path.join(folder_paths.models_dir, "tensorrt", "upscaler")

def get_default_engine():
    return os.listdir(ENGINE_DIR)[0]

class UpscalerTensorrt:
    def __init__(self):
        self.engine_instance = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "images": ("IMAGE",),
                "engine": (os.listdir(ENGINE_DIR),),
                "keep_model_loaded": ("BOOLEAN",)
            }
        }

    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "tensorrt"

    def main(self, images, engine, keep_model_loaded):
        images = images.permute(0, 3, 1, 2)  # B,C,W,H
        B, C, W, H = images.shape
        shape_dict = {
            "input": {"shape": (1, 3, W, H)},
            "output": {"shape": (1, 3, W*4, H*4)},
        }

        if not self.engine_instance or not keep_model_loaded:
            # setup TensorRT engine
            self.engine_instance = Engine(os.path.join(ENGINE_DIR, engine), keep_model_loaded=keep_model_loaded)
            self.engine_instance.load()
            self.engine_instance.activate()
            self.engine_instance.allocate_buffers(shape_dict=shape_dict)
        
        engine = self.engine_instance
        custom_stream = torch.cuda.Stream()
        cudaStream = custom_stream.cuda_stream

        pbar = ProgressBar(B)
        images_list = list(torch.split(images, split_size_or_sections=1))
        upscaled_frames = []

        # CUDA event for timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        total_start_event = torch.cuda.Event(enable_timing=True)
        total_end_event = torch.cuda.Event(enable_timing=True)

        total_start_event.record()

        for img in images_list:
            torch.cuda.nvtx.range_push("Inference")
            start_event.record()
            result = engine.infer({"input": img}, cudaStream)
            end_event.record()
            torch.cuda.synchronize()  # Ensure the event timings are correct
            inference_time = start_event.elapsed_time(end_event)
            print(f"Inference time for current image: {inference_time} ms")
            torch.cuda.nvtx.range_pop()

            output = result['output'].cpu().numpy().squeeze(0)
            output = np.transpose(output, (1, 2, 0))
            output = np.clip(255.0 * output, 0, 255).astype(np.uint8)

            upscaled_frames.append(output)
            pbar.update(1)
        
        # ensure custom stream synchronization
        custom_stream.synchronize()

        total_end_event.record()
        torch.cuda.synchronize()
        total_elapsed_time = total_start_event.elapsed_time(total_end_event)  # milliseconds
        print(f"Total inference time: {total_elapsed_time} ms")
        
        upscaled_frames_np = np.array(upscaled_frames).astype(np.float32) / 255.0
        return (torch.from_numpy(upscaled_frames_np),)

NODE_CLASS_MAPPINGS = {
    "UpscalerTensorrt": UpscalerTensorrt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UpscalerTensorrt": "Upscaler Tensorrt",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
