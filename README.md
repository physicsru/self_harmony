# Self-Harmony: Reinforcement Learning Framework

This repository contains an implementation of a reinforcement learning framework built upon the VERL (Volcano Engine Reinforcement Learning) foundation.

## Quick Start

### Prerequisites

Ensure you have the following modules loaded in your environment:

```bash
module load cuda/12.6
module load cudnn/9.4.0
module load nccl/2.24.3
```

### Installation

1. **Install dependencies:**
   ```bash
   bash scripts/install_env.sh
   ```

2. **Install the package in development mode:**
   ```bash
   pip install -e . --no-deps
   ```

### Usage

After installation, you can begin using the framework for your reinforcement learning experiments.

## Framework Overview

This implementation extends the VERL framework to provide enhanced capabilities for reinforcement learning applications, particularly focused on large language model training and optimization through self-harmony learning approaches.

### Training Data Pipeline

The framework includes tools for generating auxiliary training datasets:

- `create_train_aux.py`: Transforms original problems using creative reframing while preserving mathematical correctness
- Supports GPQA Diamond dataset format
- Generates alternative problem formulations that test the same mathematical concepts through different contexts

## Directory Structure

```
├── scripts/           # Installation and utility scripts
├── verl/             # Core framework modules
├── run_script/       # Execution scripts for different model sizes
└── setup.py          # Package configuration
```

## Requirements

- CUDA 12.6 or compatible
- cuDNN 9.4.0 or compatible
- NCCL 2.24.3 or compatible
- Python 3.10

