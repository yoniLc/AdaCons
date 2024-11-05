# AdaCons: Adaptive Consensus Gradients Aggregation for Scaled Distributed Training

Sample Implementation of the paper ["Adaptive Consensus Gradients Aggregation for Scaled Distributed Training"]().


## Abstract
Distributed machine learning has recently become a critical paradigm for training large models on vast datasets. 
We examine the stochastic optimization problem for deep learning within synchronous parallel computing environments under communication constraints.

While averaging distributed gradients is the most widely used method for gradient estimation, whether this is the optimal strategy remains an open question. 
In this work, we analyze the distributed gradient aggregation process through the lens of subspace optimization. 
By formulating the aggregation problem as an objective-aware subspace optimization problem, we derive an efficient weighting scheme for gradients, guided by subspace coefficients. 
We further introduce subspace momentum to accelerate convergence while maintaining statistical unbiasedness in the aggregation. 

Our method demonstrates improved performance over the ubiquitous gradient averaging on multiple MLPerf tasks while remaining extremely efficient in both communicational and computational complexity.

## Installation
1. Create and activate a Conda environment:
   ```bash
   conda create -n venv_adacons python=3.8
   conda activate venv_adacons
   ```
2. Install PyTorch with CUDA:
   ```bash
   conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

## Code
The code provides an illustration, using a simple linear regression example, on how to apply the proposed AdaCons gradient aggregation technique within PyTorch’s distributed data-parallel (DDP) framework.

Please adjust the configuration to match your system settings:
```bash
torchrun --nproc_per_node=$n_local_gpus --nnodes=$n_nodes --node_rank=$rank --master_addr=$master_addr --master_port=$master_port Main.py
```
Running this script (1000 iterations) give the following final loss performance.

| Effective Batch Size      | Baseline (Sum) | Adacons (32 workers) |  Adacons (8 workers) |
|-----------------|-----------------|-----------------|-----------------
| 128            | 33.4   | 1.98E-8                    | 1.27E-5   |
| 2K             | 49.3   | 1.48E-8 (iter 300)         | 52.4   |

## Reference
    @article{choukroun2024adacons,
      title={Adaptive Consensus Gradients Aggregation for Scaled Distributed Training},
      author={Choukroun, Yoni and Azoulay, Shlomi and Kisilev, Pavel},
      journal={arXiv preprint arXiv:},
      year={2024}
    }
