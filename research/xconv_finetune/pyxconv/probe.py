import torch
from typing import List
import gc

@torch.jit.script
def rand_with_zeros(N: int, ps:int, indices, X):
    """
    Random matrix (normal distribution) with mask.

    Arguments:
        N (int): Number of pixels.
        ps (int): Number of probing vectors.
        scale (int): Scaling factor due to pseudo-orthogonalization overlap.
        indices (Tensor): Non zero indices, same for all pixels.
        X (Tensor): Input tensor (untouched, used for device detection).
    """
    n = indices.shape[0]
    a = torch.zeros(N, ps, device=X.device)
    a[:, indices] = torch.randn(N, n, device=X.device)
    return torch.sqrt(ps/n)*a


@torch.jit.script
def draw_r(ps:int, ci: int, N: int, X):
    """
    Draws a simple random multi-channel probing matrix.

    Arguments:
        ps (int): Number of probing vectors.
        ci (int): Number of input channels.
        N (int): Number of pixels.
        X (tensor): Input tensor (untouched, used for device detection).
    """
    return torch.randn(ci*N, ps, device=X.device)


@torch.jit.script
def draw_o(ps:int, ci: int, N: int, X):
    """
    Draws a pseudo block-orthogonal probing matrix. For a large enough number of probing vector (ps > 8 * ci)
    this matrix will be exactly block orthogonal

    Arguments:
        ps (int): Number of probing vectors.
        ci (int): Number of input channels.
        N (int): Number of pixels.
        X (tensor): Input tensor (untouched, used for device detection).
    """
    if ps // ci > 8:
        n = ps // ci
        inds = torch.split(torch.randperm(ps, dtype=torch.long), n)
    else:
        n = 8
        inds = torch.split(torch.randperm(n*ci, dtype=torch.long) % ps, n)
    e = torch.cat([rand_with_zeros(N, ps, inds[i], X) for i in range(ci)], dim=0)
    return e


@torch.jit.script
def back_probe_f(
    N: int, # Number of pixels: 16129
    ci: int, # Number of input channels: 512
    co: int, # Number of output channels: 1000
    b: int, # batch-size eg. 10
    ps: int, # 64
    nw: int, # 1
    offs: List[int], # [tensor(0)]
    grad_output, # (b, co, nx',ny') eg. (10, 1000, 127, 127)
    eX # (ps, b, ci) eg. (64, 10, 512)
):
    """
    Backward pass of probing-based convolution filter gradient.

    Arguments:
        seed (int): Random seed for probing vectors
        N (int): Number of pixels
        ci (int): Number of input channels
        co (int): Number of output channels
        b (int): Batch size
        ps (int): Number of probing vectors
        nw (int): Convolution filter width (nw in each direction)
        offs (List): List of offsets for the convolution filter coefficients
        grad_output (Tensor): Backward input
        eX (Tensor): Forward pass probed Tensor (b x ps)

    Returns:
        gradient w.r.t convolution filter
    """
    # Init gradient
    
    # (co, ci, nw) eg. (1000, 512, 1)
    grad_weight = torch.zeros(co, ci, nw, device=eX.device, dtype=eX.dtype)
    
    # (b, co, nx*ny) eg. (10, 1000, 16129)
    Ye = grad_output.view(b, co, -1)
    dtype = Ye.dtype   # master dtype (usually float32)

    # (N, ps) eg. (16129, 64)
    e = torch.randn(N, ps, device=eX.device, dtype = dtype)
    
    # (ps, co, ci) eg. (64, 1000, 512)
    eYXe = torch.zeros(ps, co, ci, device=eX.device, dtype=eX.dtype)
    
    for i, o in enumerate(offs):
        
        # i: 0, o: tensor(0)
        
        # (N, ps) eg. (16129, 64)
        ev = e if N == 1 else e.roll(-int(o), dims=0)

        # (ps, co, b) eg. (64, 1000, 10)
        eY = Ye.matmul(ev).permute((2, 1, 0)).contiguous()

        # (ps, co, ci) eg. (64, 1000, 512)    
        torch.bmm(eY, eX, out=eYXe)
        
        # grad_weight[:, : , 0], eg. (co, ci) (1000, 512)
        grad_weight[:, :, i] += eYXe.sum(0)

    return grad_weight / ps



@torch.jit.script
def back_probe_a(N: int, ci: int, co: int, b: int, ps: int, nw: int,
                 offs: List[int], grad_output, eX):
    """
    Backward pass of probing-based convolution filter gradient.
    Arguments:
        seed (int): Random seed for probing vectors
        N (int): Number of pixels
        ci (int): Number of input channels
        co (int): Number of output channels
        b (int): Batch size
        ps (int): Number of probing vectors
        nw (int): Convolution filter width (nw in each direction)
        offs (List): List of offsets for the convolution filter coefficients
        grad_output (Tensor): Backward input
        eX (Tensor): Forward pass probed Tensor (b x ps)
    Returns:
        gradient w.r.t convolution filter
    """
    # Redraw e
    e = draw_r(ps, ci, N, eX).view(ci, N, ps)

    # Y' X e
    Ye = grad_output.view(b, -1)
    LRe = torch.mm(eX.t(), Ye).view(ps, co, N)
    # Init gradient
    grad_weight = torch.zeros(co, ci, nw, device=eX.device)

    for i, o in enumerate(offs):
        ev = e if N==1 else e.roll(-int(o), dims=1)
        grad_weight[:, :, i] = torch.einsum('bjk, lkb -> jl', LRe, ev)
    return grad_weight / ps


@torch.jit.script
def back_probe_o(N: int, ci: int, co: int, b: int, ps: int, nw: int,
                 offs: List[int], grad_output, eX):
    """
    Backward pass of probing-based convolution filter gradient.
    Arguments:
        seed (int): Random seed for probing vectors
        N (int): Number of pixels
        ci (int): Number of input channels
        co (int): Number of output channels
        b (int): Batch size
        ps (int): Number of probing vectors
        nw (int): Convolution filter width (nw in each direction)
        offs (List): List of offsets for the convolution filter coefficients
        grad_output (Tensor): Backward input
        eX (Tensor): Forward pass probed Tensor (b x ps)
    Returns:
        gradient w.r.t convolution filter
    """
    # Redraw e
    e = draw_o(ps, ci, N, eX).view(ci, N, ps)

    # Y' X e
    Ye = grad_output.view(b, -1)
    LRe = torch.mm(eX.t(), Ye).view(ps, co, N)
    # Init gradient
    grad_weight = torch.zeros(co, ci, nw, device=eX.device)

    for i, o in enumerate(offs):
        ev = e if N==1 else e.roll(-int(o), dims=1)
        grad_weight[:, :, i] = torch.einsum('bjk, lkb -> jl', LRe, ev)
    return grad_weight / ps

def comp_tensor_memory_mb(my_tensor):
    size_in_bytes = my_tensor.numel() * my_tensor.element_size()

    size_in_mb = size_in_bytes / (1024 * 1024)

    return size_in_mb

@torch.jit.script
def fwd_probe_f(
    ps: int, # 64
    b: int, # 32768 
    ci: int, # Number of input channels = 3
    N: int, # Number of pixels = 1024
    X # Input Tensor, # (B, C, H, W) eg. (32768, 3, 32, 32) # Occupies 384MB
):
    """
    Forward pass of probing-based convolution filter gradient.

    Arguments:
        ps (int): Number of probing vectors
        X (Tensor): Layer's input Tensor

    Returns:
        eX (Tensor): Probed input tensor to be saved for backward pass
    """

    # (b, ci, nx, ny) --> (b, ci, N = nx*ny) eg. (32768, 3, 1024)
    # Failure: (32768, 384, 9)
    Xv = X.view(b, ci, -1)

    # (N, ps) eg. (1024, 64)
    # Failure: (9, 64)
    e = torch.randn(N, ps, device=X.device, dtype=X.dtype)
    
    # (b, ci, ps) eg. (32768, 3, 64)
    # Failure: (32768, 384, 64) # 3072 MB!

    if Xv.dtype == torch.float16:
        e = e.half()
    eX = Xv.matmul(e)

    # (ps, b, ci) eg. (64, 32768, 3)
    # return eX.permute((2, 0, 1)).contiguous()
    return eX.permute((2, 0, 1)).contiguous()


@torch.jit.script
def fwd_probe_a(ps: int, b: int, ci: int, N: int, X):
    """
    Forward pass of probing-based convolution filter gradient.
    Arguments:
        ps (int): Number of probing vectors
        X (Tensor): Layer's input Tensor
    Returns:
        eX (Tensor): Probed input tensor to be saved for backward pass
    """
    Xv = X.reshape(b, -1)
    e = draw_r(ps, ci, N, X)

    return torch.mm(Xv, e.reshape(ci*N, ps))


@torch.jit.script
def fwd_probe_o(ps: int, b: int, ci: int, N: int, X):
    """
    Forward pass of probing-based convolution filter gradient.
    Arguments:
        ps (int): Number of probing vectors
        X (Tensor): Layer's input Tensor
    Returns:
        eX (Tensor): Probed input tensor to be saved for backward pass
    """
    Xv = X.reshape(b, -1)
    e = draw_o(ps, ci, N, X)

    return torch.mm(Xv, e.reshape(ci*N, ps))


# Access dictionaries
back_probe = {'gaussian': back_probe_a, 'orthogonal': back_probe_o, 'independent': back_probe_f}
fwd_probe = {'gaussian': fwd_probe_a, 'orthogonal': fwd_probe_o, 'independent': fwd_probe_f}
draw_e = {'orthogonal': draw_o, 'gaussian': draw_r}