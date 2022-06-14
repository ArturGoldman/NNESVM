import torch
from tqdm import tqdm


def process_cv(model, batch, operating_device,
               out_model_dim, target_dist_dim,
               grad_log,
               cr_gr=False, mode="simple_additive", to_detach=False):
    if mode == "simple_additive":
        if out_model_dim != 1:
            # model output dim should be 1 to be subtracted from f
            raise ValueError("Model output dimension is not 1")
        return model(batch)
    elif mode == "stein":
        # here we interpret model as nabla varphi
        if out_model_dim != target_dist_dim:
            raise ValueError("Model output dimension is not equal to distribution dimension")
        laplacians = []
        batch.requires_grad = True
        for x in tqdm(batch, desc="Processing stein CV"):
            jacob = torch.autograd.functional.jacobian(model, x.reshape(1, -1),
                                                       create_graph=cr_gr, vectorize=True).squeeze()
            if to_detach:
                laplacians.append(torch.trace(jacob).detach())
            else:
                laplacians.append(torch.trace(jacob))

        batch.requires_grad = False
        batch = batch.to('cpu')
        log_grads = grad_log(batch).to(operating_device)
        batch = batch.to(operating_device)
        if to_detach:
            with torch.no_grad():
                outs = model(batch)
        else:
            outs = model(batch)
        return (torch.stack(laplacians) + (log_grads*outs).sum(dim=1)).reshape(-1, 1)
    elif mode == "stein_classic":
        # here we interpret model as varphi
        # WARNING: CURRENTLY UNDER CONSTRUCTION
        if out_model_dim != 1:
            # model output dim should be 1 for defined implementation
            raise ValueError("Model output dimension is not 1. It should be while using stein_classic CV")

        grads = []
        batch.requires_grad = True
        for x in tqdm(batch, desc="Processing stein CV, grad"):
            res = model(x)
            grad = torch.autograd.grad(res, x, create_graph=cr_gr)
            if to_detach:
                grads.append(grad[0].detach())
            else:
                grads.append(grad[0])


        laplacians = []
        batch.requires_grad = True

        for x in tqdm(batch, desc="Processing stein CV, laplacian"):
            hess = torch.autograd.functional.hessian(model, x, create_graph=cr_gr)
            if to_detach:
                laplacians.append(torch.trace(hess).detach())
            else:
                laplacians.append(torch.trace(hess))

        batch.requires_grad = False
        batch = batch.to('cpu')
        log_grads = grad_log(batch).to(operating_device)
        # batch = batch.to(operating_device)
        return (torch.stack(laplacians) + (log_grads*torch.stack(grads)).sum(dim=1)).reshape(-1, 1)
    else:
        raise ValueError("Unrecognised CV type")


