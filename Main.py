import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext

class AdaConsAggregator():
    def __init__(self, model, momentum=0.):
        self.model = model
        self.world_size = dist.get_world_size()

        self.eps = 1e-8
        self.momentum = momentum
        self.momentum_alphas = None

        self.model.register_comm_hook(state=None, hook=self._adacons_hook)

        self.reset()
            
    def _adacons_hook(self, state, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
        g_i = bucket.buffer().clone()
        def update_state(fut):
            g_sum: torch.Tensor = fut.value()[0]
            self.dot_i += (g_sum*g_i).sum()
            self.norm_g_i += (g_i**2).sum()
            return g_i
        return dist.all_reduce(bucket.buffer(), async_op=True).get_future().then(update_state)

    def reset(self):
        self.dot_i = 0.
        self.norm_g_i = 0.

    def aggregate(self):
        alpha_i = self.dot_i/(torch.sqrt(self.norm_g_i) + self.eps)/self.world_size
        # Allgather subspace coefficients
        alphas = [torch.zeros(1, device=self.dot_i.device) for _ in range(self.world_size)]
        dist.all_gather(alphas, alpha_i)
        alphas = torch.cat(alphas)
        # Momentum
        sorted_alphas = torch.sort(alphas)
        self.momentum_alphas = sorted_alphas[0] if self.momentum_alphas is None else self.momentum * self.momentum_alphas + (1 - self.momentum) * sorted_alphas[0]
        alphas = self.momentum_alphas[sorted_alphas[1]]
        # Unbias estimation
        alphas = alphas/alphas.sum()
        alpha_i = (alphas[dist.get_rank()] if self.world_size > 1 else alphas) / (torch.sqrt(self.norm_g_i) + self.eps)
        # alpha_i = 1./self.world_size # SUM (averaging) baseline
        ###
        handles = [] 
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(alpha_i)
                handles.append(dist.all_reduce(param.grad.data, async_op=True))

        for h in handles:
            h.wait()
        self.reset()

if __name__ == '__main__':
    def set_seed(seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True 
    set_seed()
    world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    ###
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    ###
    class LinearModel(nn.Module):
        def __init__(self, dim=1000):
            super(LinearModel, self).__init__()
            self.linear = nn.Linear(dim, 1, bias=False)
            with torch.no_grad():
                self.linear.weight.data = -5+10*torch.rand(self.linear.weight.shape) # Initial distance from optimum
        def forward(self, x):
            return self.linear(x)
    ###
    dim = 1000
    ddp_model = DDP(LinearModel(dim).to(device), device_ids=[int(os.environ['LOCAL_RANK'])])
    set_seed(int(os.environ["RANK"])) # for random data generation
    ###
    optimizer = optim.SGD(ddp_model.parameters(), lr=1e-4)
    optimizer.aggregator = AdaConsAggregator(ddp_model, momentum=0.99)
    # optimizer.aggregator = lambda: None;optimizer.aggregator.aggregate = lambda *args: None # Dummy aggregator for baseline
    ###
    bs = 2048//world_size
    accumulation_steps = 2
    num_iters = 1000
    for iter in range(1, num_iters+1):
        with ddp_model.no_sync() if iter%accumulation_steps != 0 else nullcontext():
            pred = ddp_model(torch.rand(bs,dim).to(device))
            loss = 0.5*(pred**2).sum(-1).mean()
            loss.backward()
            if iter%accumulation_steps == 0:
                optimizer.aggregator.aggregate()   
                # optimal step size:
                for g in optimizer.param_groups:
                    g['lr'] = (ddp_model.module.linear.weight.grad*ddp_model.module.linear.weight.data).sum()/(ddp_model.module.linear.weight.grad.norm()**2)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if iter % 100 == 0 and dist.get_rank()==0:
                print(f"Iter: {iter}, Loss: {loss.item():4e}")

    dist.destroy_process_group()
