import torch  
from pathlib import Path  
from yolo.model.yolo import create_model  
from yolo.tools.solver import InferenceModel
from yolo.config.config import Config  
from hydra import compose, initialize  
from torch.onnx import export  
  
# Load configuration  
with initialize(config_path='yolo/config', version_base=None):
    cfg = compose(config_name='config', overrides=['model=v9-c', 'dataset=ppe_500'])
  
# Create and load model  
# model = create_model(cfg.model, class_num=4, weight_path='epoch_0.ckpt')  
model= InferenceModel.load_from_checkpoint("epoch_0.ckpt", cfg) 
model.eval()  
  
# Create dummy input matching your image size  
dummy_input = torch.ones((1, 3, 640, 640))  
  
# Export to ONNX  
output_path = "model.onnx"  
export(  
    model,  
    dummy_input,  
    output_path,  
    input_names=["input"],  
    output_names=["output"],  
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  
)  
print(f"ONNX model saved to {output_path}")