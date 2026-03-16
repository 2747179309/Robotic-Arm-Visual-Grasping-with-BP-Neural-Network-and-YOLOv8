import torch
from ultralytics import YOLO
model = YOLO('bests.pt')
model.model.eval()  
dummy_input = torch.randn(1, 3, 480, 640)
torch.onnx.export(
model.model,
dummy_input,
'bests.onnx',
export_params=True,
opset_version=11,
do_constant_folding=True,
input_names=['input'],
output_names=['output'],
dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

print("Successful!")
