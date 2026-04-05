"""
Measure inference time for a model across different sequence lengths.
"""

import time
import torch
import numpy as np
from hydrapfn.scripts.model_loader import load_model


# Configuration
MODEL_PATH = "hydrapfn/trained_models/inference_test.cpkt"
DEVICE = "cuda"
NUM_CLASSES = 10
NUM_FEATURES = 20
SEQ_LENGTHS = [5000, 10000, 20000]
NUM_RUNS = 3
WARMUP = 1


def create_mock_data(seq_length, num_features, num_classes, device="cpu", dtype=torch.float32):
    """Create mock input data for inference."""
    x_data = torch.randn(seq_length, 1, num_features, device=device, dtype=dtype)
    y_data = torch.randint(0, num_classes, (seq_length, 1), device=device, dtype=torch.long)
    return x_data, y_data


def measure_inference_time(model, seq_length, num_features, num_classes, device="cpu", num_runs=3, warmup=1):
    # Infer dtype from model parameters — handles fp16, bf16, fp32 automatically
    model_dtype = next(model.parameters()).dtype

    x_data, y_data = create_mock_data(seq_length, num_features, num_classes, device, dtype=model_dtype)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model((None, x_data, y_data.float()), single_eval_pos=seq_length - 1)
    
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            start = time.time()
            _ = model((None, x_data, y_data.float()), single_eval_pos=seq_length - 1)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
    
    return np.mean(times), np.std(times)


def main():
    print(f"Loading model from {MODEL_PATH}...")
    model, optimizer, config = load_model(MODEL_PATH, device=DEVICE)
    model.eval()
    print(f"Model type: {config.get('model_type', 'unknown')}")
    print()
    
    print(f"{'Seq Length':<15} {'Avg Time (ms)':<20} {'Std Dev (ms)':<20} {'Throughput (samples/s)':<20}")
    print("-" * 75)
    
    results = []
    for seq_length in SEQ_LENGTHS:
        try:
            avg_time, std_time = measure_inference_time(
                model, seq_length, NUM_FEATURES, NUM_CLASSES,
                device=DEVICE, num_runs=NUM_RUNS, warmup=WARMUP
            )
            
            avg_ms = avg_time * 1000
            std_ms = std_time * 1000
            throughput = seq_length / avg_time
            
            print(f"{seq_length:<15} {avg_ms:<20.2f} {std_ms:<20.2f} {throughput:<20.2f}")
            results.append({'seq_length': seq_length, 'avg_ms': avg_ms, 'throughput': throughput})
            
        except Exception as e:
            print(f"Error at seq_length {seq_length}: {e}")
    
    print("-" * 75)


if __name__ == "__main__":
    main()

