import torch

def gradient_accumulation(w_init: torch.Tensor, micro_batches: list, lr: float, accum_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (final_weights, last_mean_gradient) as float32 tensors.
    """

    w = w_init.clone().detach().requires_grad_(True)

    last_mean_gradient = torch.zeros_like(w)

    # Process groups of accum_steps micro-batches
    for start in range(0, len(micro_batches), accum_steps):

        # Clear gradients at the beginning of each group
        w.grad = None

        # Accumulate gradients from K micro-batches
        for i in range(start, start + accum_steps):
            x, t = micro_batches[i]

            # L = (w^T x - t)^2
            loss = (w @ x - t) ** 2

            # Accumulate gradient in w.grad
            loss.backward()

        # Average the accumulated gradient
        last_mean_gradient = w.grad / accum_steps

        # Update weights without tracking the update in autograd
        with torch.no_grad():
            w -= lr * last_mean_gradient

    return w.float(), last_mean_gradient.float()
    
       
       
       

   