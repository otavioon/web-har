# export_onnx.py
import torch
from convnet import ConvNet

model = ConvNet()
model.eval()

dummy = torch.randn(1, 6, 60)
model_output = model(dummy)
print("Model output shape:", model_output.shape)

print("🚀 Exporting to JIT format...")
traced = torch.jit.trace(model, torch.randn(1, 6, 60))
traced.save("convnet.pt")

print("✅ Exported convnet.pt")
