# Installing a Refusal Circuit into a Language Model

In this experiment, I rewrote a set of weights in a Qwen model to predict the exact sequence ```I'm sorry, I cannot assist with this request.``` A pair of MLP neurons in the same model is trained to serve as a harmfulness detector: if the detector fires, the input is routed through the circuit and the model generates the refusal sequence; otherwise, the input is routed through the remainder of the model and generations remain (essentially) intact.

With the installed circuit, the base model `Qwen2.5-1.5B` nearly matches the refusal rates of the safety-trained `Qwen2.5-1.5B-Instruct`, at a perplexity cost of only 0.4% relative to the base model without the circuit. Even more importantly, this approach to refusal is **interpretable by construction**: if the model under- or over-refused, we can see precisely what went wrong.

<p align="center">
  <img src="https://github.com/user-attachments/assets/bd5fb209-e18a-4d35-a471-f8e0a54c95b0" alt="Refusal Circuit" width="600">
</p>

## Quick Start

```bash
# Clone the repo
git clone https://github.com/dariakryvosheieva/refusal-circuit.git
cd refusal-circuit

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the end-to-end experiment
sbatch --export=ALL,EXTRA_ARGS="--save-generations generations/gen.jsonl" job.sbatch

# Run the detector layer sweep
sbatch sweep.sbatch
```

## Repo Structure

```
refusal-circuit/
├── data/                              # Training and evaluation datasets
├── architecture.py                    # Architectural facts and addressing helpers
├── data.py                            # Data loading utilities
├── evaluation.py                      # Evaluation utilities
├── job.sbatch                         # Shell script to submit the main experiment on GPU
├── judge_generations.py               # Judging script to score responses for refusal and harmful content
├── refusal_circuit.py                 # Library for circuit training and installation
├── run_refusal_circuit.py             # Driver for the main experiment
├── sweep_detector_layer.py            # Detector layer sweep
└── sweep.sbatch                       # Shell script to submit the sweep on GPU
```