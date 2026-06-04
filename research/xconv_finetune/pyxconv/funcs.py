import torch
import torch.nn.functional as F

from pyxconv.utils import *
from pyxconv.probe import *


def _fwd_probe_grouped(mode, ps, b, ci, co, groups, nx, ny, input):
    """Probe input for XConv; for groups>1, probe each group separately."""
    if groups == 1:
        return fwd_probe[mode](ps, b, ci, nx * ny, input)
    ci_pg = ci // groups
    co_pg = co // groups
    if ci_pg * groups != ci or co_pg * groups != co:
        raise ValueError(
            f"in_channels={ci} and out_channels={co} must be divisible by groups={groups}"
        )
    parts = []
    for g in range(groups):
        sl = slice(g * ci_pg, (g + 1) * ci_pg)
        parts.append(fwd_probe[mode](ps, b, ci_pg, nx * ny, input[:, sl]))
    return torch.cat(parts, dim=2)


def _back_probe_grouped(
    mode, ps, b, ci, co, groups, nx, ny, nw_k, offs, delta, eX, seed
):
    """Weight gradient for grouped XConv; matches shape (co, ci//groups, k, k)."""
    n_off = len(offs)
    if groups == 1:
        with random_seed_torch(int(seed)):
            return back_probe[mode](
                nx * ny, ci, co, b, ps, n_off, offs, delta, eX
            )
    ci_pg = ci // groups
    co_pg = co // groups
    dw = torch.zeros(co, ci_pg, n_off, device=eX.device, dtype=eX.dtype)
    with random_seed_torch(int(seed)):
        for g in range(groups):
            sl_in = slice(g * ci_pg, (g + 1) * ci_pg)
            sl_out = slice(g * co_pg, (g + 1) * co_pg)
            dw_g = back_probe[mode](
                nx * ny,
                ci_pg,
                co_pg,
                b,
                ps,
                n_off,
                offs,
                delta[:, sl_out],
                eX[:, :, sl_in],
            )
            dw[sl_out] = dw_g
    return dw


class Xconv2D(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, 
        input, # (B, C, H, W) eg. (10, 3, 2048, 2048)
        weight, # (num_channels, C, H, W) eg. (96, 3, 7, 7)
        ps=8, # 64
        mode='all', # independent
        bias=None, # int, 96
        stride=1, # (t, t) eg. (2, 2) 
        padding=0, # (t, t) eg. (0, 0)
        dilation=1, # (t, t) eg. (1, 1)
        groups=1
    ):
        seed = torch.randint(100000, (1,))
        
        # b = 11, ci = 3, nx = 2048, ny = 2048
        b, ci, nx, ny = input.shape
        co = weight.shape[0]
        with random_seed_torch(int(seed)):
            with torch.autograd.grad_mode.no_grad():
                eX = _fwd_probe_grouped(
                    mode, ps, b, ci, co, groups, nx, ny, input
                )

        ctx.xshape = input.shape
        ctx.stride = stride
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.padding = padding
        ctx.mode = mode
        ctx.ps = ps

        with torch.autograd.grad_mode.no_grad():
            
            # (b, co, nx', ny') eg. (10, 96, 1021, 1021)
            Y = F.conv2d(
                input, 
                weight, 
                bias=bias,
                stride=stride,
                padding=padding, 
                groups=groups
            )

        ctx.save_for_backward(eX, seed, weight, bias)

        with torch.autograd.grad_mode.no_grad():
            return Y

    @staticmethod
    def backward(ctx, grad_output):
        
        # grad_output: (B, co, nx', ny') eg. (10, 1000, 127, 127)
        # eX: (ps, b, ci) eg. (64, 10, 512)
        # weight: (co, b, nw, nw) eg. (1000, 512, 1, 1)
        # bias: (co) eg. (1000)
        eX, seed, weight, bias = ctx.saved_tensors

        dw = None
        if ctx.needs_input_grad[1]:
            nw_k = weight.shape[2]

            # b = 10, ci = 512, nx = 127, ny = 127
            b, ci, nx, ny = ctx.xshape
            co = grad_output.shape[1] # 1000

            offs = offsets2d((nx, ny), nw_k)

            # (B, co, nx', ny') eg. (10, 1000, 127, 127)
            delta = dilate2d(grad_output, co, (nx, ny), b, ctx.stride)
            with random_seed_torch(int(seed)):
                with torch.autograd.grad_mode.no_grad():
                    dw = _back_probe_grouped(
                        ctx.mode,
                        ctx.ps,
                        b,
                        ci,
                        co,
                        ctx.groups,
                        nx,
                        ny,
                        nw_k,
                        offs,
                        delta,
                        eX,
                        seed,
                    )
                ci_pg = ci // ctx.groups
                kh, kw = weight.shape[2], weight.shape[3]
                n_w = kh * kw
                if dw.shape[2] > n_w:
                    dw = dw[:, :, :n_w]
                dw = dw.reshape(co, ci_pg, kh, kw)

        dx = None
        if ctx.needs_input_grad[0]:
            
            # (b, ci, nx', ny'), eg. (10, 512, 127, 127)
            dx = torch.nn.grad.conv2d_input(
                ctx.xshape, 
                weight, 
                grad_output,
                stride=ctx.stride, 
                padding=ctx.padding,
                dilation=ctx.dilation, 
                groups=ctx.groups
            )

        db = None
        if bias is not None and ctx.needs_input_grad[4]:
            
            # (co) eg. (1000)
            db = grad_output.sum((0, 2, 3))
            

        return dx, dw, None, None, db, None, None, None, None


class Xconv3D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, ps=8, mode='all', bias=None, stride=1,
                padding=0, dilation=1, groups=1):
        seed = torch.randint(100000, (1,))
        b, ci, nx, ny, nz = input.shape
        with random_seed_torch(int(seed)):
            with torch.autograd.grad_mode.no_grad():
                eX = fwd_probe[mode](ps, b, ci, nx*ny*nz, input)

        ctx.xshape = input.shape
        ctx.stride = stride
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.padding = padding
        ctx.mode = mode
        ctx.ps = ps

        with torch.autograd.grad_mode.no_grad():
            Y = F.conv3d(input, weight, bias=bias, stride=stride,
                         padding=padding, groups=groups)

        ctx.save_for_backward(eX, seed, weight, bias)

        with torch.autograd.grad_mode.no_grad():
            return Y

    @staticmethod
    def backward(ctx, grad_output):
        eX, seed, weight, bias = ctx.saved_tensors

        dw = None
        if ctx.needs_input_grad[1]:
            nw = weight.shape[2]
            b, ci, nx, ny, nz = ctx.xshape
            co = grad_output.shape[1]

            offs = offsets3d((nx, ny, nz), nw)
            delta = dilate3d(grad_output, co, (nx, ny, nz), b, ctx.stride)
            with random_seed_torch(int(seed)):
                dw = back_probe[ctx.mode](nx*ny*nz, ci, co, b, ctx.ps,
                                          nw**3, offs, delta, eX)
            dw = dw.reshape(co, ci, nw, nw, nw)

        dx = None
        if ctx.needs_input_grad[0]:
            dx = torch.nn.grad.conv3d_input(ctx.xshape, weight, grad_output,
                                            stride=ctx.stride, padding=ctx.padding,
                                            dilation=ctx.dilation, groups=ctx.groups)

        db = None
        if bias is not None and ctx.needs_input_grad[4]:
            db = grad_output.sum((0, 2, 3, 4))

        return dx, dw, None, None, db, None, None, None, None


class Brelu(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, inplace=False):
        with torch.autograd.grad_mode.no_grad():
            Y = F.relu(input, inplace=inplace)
            sx = (Y > 0).byte()
        ctx.save_for_backward(sx)

        return Y

    @staticmethod
    def backward(ctx, grad_output):
        binp, = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            return grad_output*binp, None
        return None, None