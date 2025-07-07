import jax 
import jax.numpy as jnp

def pad_replicate(x):
    # Equivalent to F.pad(..., (1,1,1,1), mode='replicate') for 3D inputs (N, H, W)
    pad_width = ((0, 0), (1, 1), (1, 1))
    return jnp.pad(x, pad_width, mode='edge')

def avg_pool2d(x, kernel_size, stride=(1,1), padding=(0,0), divisor_override=None):
    """
    Mimics PyTorch's F.avg_pool2d with asymmetric padding and divisor_override support.
    
    Args:
        x: (N, C, H, W) or (N, H, W)
        kernel_size: tuple (kh, kw)
        stride: tuple (sh, sw)
        padding: tuple (ph, pw) — symmetric padding for top-bottom and left-right.
        divisor_override: int or None
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    pad_top, pad_bottom = padding[0], padding[0]
    pad_left, pad_right = padding[1], padding[1]
    
    if x.ndim == 4:
        # (N, C, H, W)
        pad_width = (
            (0, 0),  # batch
            (0, 0),  # channels
            (pad_top, pad_bottom),  # height
            (pad_left, pad_right)   # width
        )
        window_shape = (1, 1, kernel_size[0], kernel_size[1])
        window_strides = (1, 1, stride[0], stride[1])
    elif x.ndim == 3:
        # (N, H, W)
        pad_width = (
            (0, 0),  # batch
            (pad_top, pad_bottom),  # height
            (pad_left, pad_right)   # width
        )
        window_shape = (1, kernel_size[0], kernel_size[1])
        window_strides = (1, stride[0], stride[1])
    else:
        raise ValueError(f"Unsupported input shape {x.shape}")
    
    x_padded = jnp.pad(x, pad_width, mode='constant', constant_values=0.0)
    
    summed = jax.lax.reduce_window(
        x_padded,
        0.0,
        jax.lax.add,
        window_shape,
        window_strides,
        'VALID'
    )
    
    if divisor_override is not None:
        pooled = summed / divisor_override
    else:
        pooled = summed / (kernel_size[0] * kernel_size[1])
    
    return pooled