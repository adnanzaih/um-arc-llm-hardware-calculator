def get_reference_information():
    """Return reference information as a markdown string."""
    return """
### Training vs. Inference Mode Explained:
- **Inference**: Running a pre-trained model to generate predictions or text
- **Training**: Updating model weights to learn from data, requiring much more memory
- **Key Differences**:
  - Training requires storing optimizer states, gradients, and full activations
  - Inference primarily needs model weights and KV cache
  - Training memory usage scales with batch size and sequence length

### Batch Size and Gradient Accumulation:
- **Batch Size**: Number of samples processed simultaneously
- **Gradient Accumulation**: Number of forward/backward passes before weight update
- **Memory Impact**: Larger batch sizes require more memory for activations
- **Performance Impact**: Larger effective batch sizes (batch_size × grad_accum) often improve training stability

### KV Cache Explained:
- **What it is**: Storage of previously computed attention keys (K) and values (V) during inference
- **Why it matters**: Prevents recalculating attention for previous tokens when generating new tokens
- **Scaling**: Grows linearly with context length and model layers
- **Formula**: KV Cache Size ≈ 2 * num_layers * seq_length * hidden_size * bytes_per_element
- **Memory impact**: For long contexts (32K+), KV cache can require more memory than the model weights

### Notes on Quantization:
- **Q2-Q8**: Refers to bits per weight in quantized models
- **FP16/FP32**: Full precision floating point (16/32 bits per weight)
- **GPTQ/AWQ**: Advanced quantization methods (typically ~4 bits per weight)

### Context Length Impact:
- **Memory Usage**: Longer contexts require more VRAM for KV cache storage
- **Performance**: Very long contexts can significantly reduce inference speed
- **Tradeoffs**: Higher context enables processing more text but increases resource requirements

### CUDA Support Impact:
- **With CUDA**: Enables GPU acceleration but may require ~10% additional memory for CUDA-specific optimizations
- **Without CUDA**: May use slightly less memory but performance will be significantly slower
- CUDA support is essential for practical inference with larger models

### Tokens Per Second (TPS) Performance:
- **40+ TPS**: Excellent (real-time conversation)
- **20-40 TPS**: Very good performance
- **10-20 TPS**: Good performance
- **5-10 TPS**: Adequate for most use cases
- **1-5 TPS**: Noticeable lag in responses
- **<1 TPS**: Very slow, not suitable for interactive applications

### Hardware Recommendations:
- Models under 10B parameters can run on consumer GPUs with appropriate quantization
- 10-70B models typically require high-end GPUs or multiple GPUs
- 100B+ models generally require multiple professional GPUs or specialized hardware
- Training requires 2-8x more memory than inference for the same model

### References:
- These calculations are approximations and actual requirements may vary
- KV cache calculations assume standard attention mechanisms
- Performance estimates assume standard inference settings
- Actual performance may vary based on specific hardware, software optimizations, and batch sizes
"""
