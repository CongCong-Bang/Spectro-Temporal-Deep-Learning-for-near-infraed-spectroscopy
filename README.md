## Description

Our package introduces Physics-Informed Neural Networks (PINNs) implemented using PyTorch. The standout feature is the incorporation of CUDA Graphs
and JIT Compilers (TorchScript) for compiling models, resulting in significant performance gains up to 9x compared to the original TensorFlow v1 implementation.

## Installation

Spectro-Temporal-Deep-Learning requires following dependencies to be installed:

- [PyTorch](https://pytorch.org) >=2.0.0
- [PyTorch Lightning](https://lightning.ai/) >= 2.0.0
- [Hydra](https://hydra.cc/docs/intro/) >= 1.3

If you intend to introduce new functionalities or make code modifications, we suggest duplicating the repository and setting up a local installation:

```bash
git clone https://github.com/CongCong-Bang/Spectro-Temporal-Deep-Learning-for-near-infrared-spectroscopy
cd Spectro-Temporal-Deep-Learning-for-near-infrared-spectroscopy

# [OPTIONAL] create conda environment
conda create -n STDL python=3.10
conda activate STDL

# install package
pip install -e .
```

## Data

The data located on the server and will be downloaded automatically upon running each example.

## Quick start

Explore a variety of implemented examples within the [examples](examples) folder. To run a specific code, such as the one for the Navier-Stokes PDE, you can use:

```bash
python examples/navier_stokes/train.py
```

You can train the model using a specified configuration, like the one found in [examples/navier_stokes/configs/config.yaml](examples/navier_stokes/configs/config.yaml). Parameters can be overridden directly from the command line. For instance:

```bash
python examples/navier_stokes/train.py trainer.max_epochs=20 n_train=3000
```

To utilize our package, there are two primary options:

- Implement your training structures using Hydra, as illustrated in our provided examples.
- Directly incorporate our package to solve your custom problem.

For a practical guide on directly using our package to solve the Schrödinger PDE in a continuous forward problem, refer to our tutorial here: [tutorials/0-Schrodinger.ipynb](tutorials/0-Schrodinger.ipynb).

## Contributing

As this is the first version of our package, there might be scope for enhancements and bug fixes. We highly value community contributions. 
If you find any issues, missing features, or unusual behavior during your usage of this library, please feel free to open an issue or submit a pull request on GitHub. 
For any queries, suggestions, or feedback, please send them to [ru2689385126@163.com].

## License

Distributed under the terms of the \[MIT\] license, "Spectro-Temporal-Deep-Learning" is free and open source software.

## Citation
These algorithms come from the unpublished paper "Spectro-Temporal Deep Learning Integrated with Real-Time Near-Infrared Spectroscopy for Dynamic
Tracking of Bioactive Constituents in Chrysanthemum Extraction."
