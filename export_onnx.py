"""
Export the ConvNet PyTorch model to ONNX format for browser inference (ONNX.js).
"""

import torch
from convnet import ConvNet

# Parameters
OUTPUT_PATH = "convnet.onnx"
INPUT_SHAPE = (1, 6, 60)  # [batch, channels, time]
NUM_CLASSES = 6           # sit, stand, walk, climb, descend, run

def main():
    print("🚀 Initializing model...")
    model = ConvNet(input_shape=(6, 60), num_classes=NUM_CLASSES)
    model.eval()

    # Dummy input (for tracing)
    dummy_input = torch.randn(INPUT_SHAPE)

    # Forward pass check
    with torch.no_grad():
        output = model(dummy_input)
    print(f"✅ Model output shape: {output.shape}")

    # Export to ONNX
    print("📦 Exporting model to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=11
    )

    print(f"🎉 Export complete: {OUTPUT_PATH}")
    print("✅ You can now load it in the browser using ONNX.js!")

if __name__ == "__main__":
    main()
