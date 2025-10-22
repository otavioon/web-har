# convnet.py
import torch
import torch.nn as nn

INPUT_SHAPE = (1, 6, 60)  # [batch, channels, time]
NUM_CLASSES = 6           # sit, stand, walk, climb, descend, run
OUTPUT_PATH = "models/convnet.onnx"

class ConvNet(nn.Module):
    def __init__(self, input_shape=(6, 60), num_classes=6):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.backbone = nn.Sequential(
            nn.Conv1d(input_shape[0], 64, 5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        x = self.head(x)
        return x

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

if __name__ == "__main__":
    main()

