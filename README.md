# MetaST: Once-for-All Spatio-Temporal Forecasting via Anchor-Based Meta-Learning

## Overview

MetaST is a novel meta-learning framework for **Inductive Spatio-Temporal Forecasting with Limited Training Data (ISTF-LTD)**. Unlike existing methods that require costly fine-tuning or synthetic data augmentation, MetaST enables **once-for-all generalization** to arbitrary unseen nodes without further training.

### Key Features

- **Zero Fine-tuning**: Direct forecasting for new nodes without additional training
- **Anchor-based Context Encoding**: Captures both local and global spatio-temporal dependencies
- **Meta-learning Framework**: Treats forecasting as a meta-task over spatial dimensions
- **Strong Generalization**: Reduces RMSE by 3.94%-22.42% over state-of-the-art methods

## Architecture

MetaST consists of three core modules:

1. **Node Encoder**: Extracts temporal features from accessible and target nodes
2. **Anchor Encoder**: Generates anchor vectors by modeling spatio-temporal context through:
   - Anchor subgraphs for local spatial structure
   - Similarity-based neighbor selection
   - Graph neural networks for feature aggregation
3. **Meta-Forecaster**: Performs adaptive forecasting via closed-form ridge regression

## Dataset

The framework supports multiple spatio-temporal datasets:

- **PEMS03/PEMS04/PEMS08**: Traffic flow datasets
- Custom datasets following the same format

Datasets should be placed in the `data/` directory with the following structure:
```
data/
├── PEMS03/
├── PEMS04/
├── PEMS08/
```

## Usage

### Quick Start

1. **Build experiment configuration**:
```bash
# Build a single config
make build config=experiments/configs/metast_010/MetaST_PEMS04_01.gin

# Or build all configs
make build-all path=experiments/configs
```

2. **Run experiments**:
```bash
# Run the experiment
python -m experiments.forecast --config_path=experiment_storage/experiment_name/config.gin run

# Or use Makefile
make run command=experiment_storage/experiment_name/command
```

3. **Monitor training**:
```bash
tensorboard --logdir experiment_storage/
```

### Configuration

Experiments are configured using Gin configuration files. Example configuration:

```python
# Basic settings
build.experiment_name = 'MetaST_PEMS04'
build.module = 'experiments.forecast'
build.data_path = 'PEMS04'
build.model_type = 'metast'
build.variables_dict = { 'seed': [42, 123, 456] }

# Model configuration
metast.hidden_dims = 64
metast.dropout = 0.1

# Dataset configuration
ColdStartForecastDataset.scale = True
ColdStartForecastDataset.horizon_len = 12
ColdStartForecastDataset.lookback_mult = 1
ColdStartForecastDataset.seen_node_ratio = 0.1

# Training configuration
train.loss_name = 'mse'
train.epochs = 50
get_optimizer.lr = 1e-2
get_optimizer.lambda_lr = 1.
get_optimizer.weight_decay = 0.
get_scheduler.warmup_epochs = 5
get_data.batch_size = 64
Checkpoint.patience = 7
```

### Model Configuration

The MetaST model can be configured with various components:

```python
# Node encoder options
metast.node_encoder_config = {
    'input_dims': 3,
    'use_pretrained_encoder': False,
    'pretrained_config': {
    },
    'ne_layer_num': 3,
    'activation': 'relu'
}

# Anchor encoder options
metast.anchor_encoder_config = {
    'feature_num': 3,
    'node_num': 30,  # e.g., int(307 * 0.1) for PEMS04
    'k_neighbors': 5,
    'gnn_layer_num': 2,
    'gnn_use_residual': True,
    'use_bias': True,
    'learn_eps': False,
    'mlp_layer_num': 2,
    'activation': 'relu'
    'sim_metric': 'attention',
    'gnn_type': 'gcn',
    'aggregation_type': 'pool_gating',
}

# Predictor options
metast.predictor_config = {
    'x_len': 12,
    'y_len': 12,
    'inr_layer_num': 3,
    'n_fourier_feats': 1024,
    'scales': [0.01, 0.1, 1, 5, 10, 20, 50, 100],
    'activation': 'relu',
    'mlp_layer_num': 3,
    'gru_layers': 2,
    'bidirectional': False,
    'num_heads': 8,
    'condition_type': 'film'
}
```

## Project Structure

```
MetaST/
├── data/
├── datasets/
├── forecaster/
│   ├── Makefile
│   ├── scripts.txt
│   ├── experiments/
│   │   ├── configs/
│   │   ├── base.py
│   │   ├── checkResult.py
│   │   ├── datasets.py
│   │   └── forecast.py
│   ├── models/
│   │   ├── deeptime/
│   │   ├── koopa/
│   │   ├── metast/
│   │   └── nbeats/
│   ├── utils/
└── └── experiment_storage/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
