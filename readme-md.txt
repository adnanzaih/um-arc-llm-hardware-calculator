# LLM Hardware Requirements Calculator

A Streamlit application to estimate hardware requirements for running and training Large Language Models (LLMs).

## Features

- Calculate VRAM requirements based on model size, quantization, and context length
- Compare inference vs. training mode requirements
- Visualize memory breakdown for different components (weights, KV cache, optimizer states, etc.)
- Compare hardware options from consumer GPUs to professional data center solutions
- Estimate performance in tokens per second
- Advanced analysis of parameter vs. memory scaling
- Training options including batch size, gradient accumulation, and optimizer selection

## Installation

1. Clone this repository
```
git clone https://github.com/yourusername/llm-hardware-calculator.git
cd llm-hardware-calculator
```

2. Create a virtual environment (optional but recommended)
```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies
```
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```
streamlit run app.py
```

The application will open in your default web browser.

## Project Structure

- `app.py`: Main application entry point
- `ui.py`: User interface components and display functions
- `calc.py`: Calculation logic for memory requirements and performance estimates
- `info.py`: Reference information and documentation

## License

MIT

## Acknowledgements

- Created for educational purposes to help understand the hardware requirements of large language models
- Memory usage formulas are approximations based on common architectures and may vary for specific models
