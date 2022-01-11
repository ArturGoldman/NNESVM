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
        if out_model_dim != target_dist_dim:
            raise ValueError("Model output dimension is not equal to distribution dimension")
        laplacians = []
        batch.requires_grad = True
        for x in tqdm(batch, desc="Processing stein CV"):
            jacob = torch.autograd.functional.jacobian(model, x, create_graph=cr_gr, vectorize=True)
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
        return torch.stack(laplacians) + (log_grads*outs).sum(dim=1)
    else:
        raise ValueError("Unrecognised CV type")


