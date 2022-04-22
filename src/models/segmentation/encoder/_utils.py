import torch
import torch.nn as nn

import numpy as np


def patch_first_conv(model, in_channels):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            break

    # change input channels for first conv
    module.in_channels = in_channels
    weight = module.weight.detach()
    reset = False

    if in_channels == 1:
        weight = weight.sum(1, keepdim=True)
    elif in_channels == 2:
        weight = weight[:, :2] * (3.0 / 2.0)
    else:
        reset = True
        weight = torch.Tensor(
            module.out_channels,
            module.in_channels // module.groups,
            *module.kernel_size
        )

    module.weight = nn.parameter.Parameter(weight)
    if reset:
        module.reset_parameters()


def replace_strides_with_dilation(module, dilation_rate):
    """Patch Conv2d modules replacing strides with dilation"""
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            mod.stride = (1, 1)
            mod.dilation = (dilation_rate, dilation_rate)
            kh, kw = mod.kernel_size
            mod.padding = ((kh // 2) * dilation_rate, (kh // 2) * dilation_rate)

            # Kostyl for EfficientNet
            if hasattr(mod, "static_padding"):
                mod.static_padding = nn.Identity()


# %%
def pad_tensor_list_equal(tensor_list, combine=0):
    
    is_tuple = isinstance(tensor_list, tuple)
    if is_tuple:
        tensor_list = list(tensor_list)
    n_items = len(tensor_list)
    n_dims = len(tensor_list[0].shape)

    dims = np.zeros((n_items, n_dims))
    max_dim = np.zeros((1, n_dims))
    for i in range(n_items):
        dims[i, :] = np.array(tensor_list[i].shape)
    max_dim = np.max(dims, axis=0)


    for i in range(n_items):
        delta_dim = np.abs(dims[i, :] - max_dim)
        pad_dims = []

        if isinstance(tensor_list[0], torch.Tensor):
            for j in reversed(range(n_dims)):
                tem = delta_dim[j]//2
                pad_dims.append(int(delta_dim[j] - tem))
                pad_dims.append(int(tem))
            tensor_list[i] = torch.nn.functional.pad(tensor_list[i], pad_dims, 'constant', 0)

        elif isinstance(tensor_list[0], torch.Tensor):
            for j in range(n_dims):
                tem = (delta_dim[j]//2, delta_dim[j] - delta_dim[j]//2)
                pad_dims.append(tem)
            tensor_list[i] = np.pad(tensor_list[i], pad_dims, 'constant')

        else:
            print("Data should be torch.Tensor or numpy.array")
            return


    # if A and B are of shape (3, 4), 
    # torch.cat([A, B], dim=0) will be of shape (6, 4)
    # torch.stack([A, B], dim=0) will be of shape (2, 3, 4)
    if combine == 0 and is_tuple:
            tensor_list = tuple(tensor_list)
    elif combine == 1:
        tensor_list = torch.cat(tensor_list, dim=0)
        # print('cat')
    elif combine == 2:
        tensor_list = torch.stack(tensor_list, dim=0)
        # print('stack)
    
    return tensor_list



# %%

def pad_array_list(array_list, pad_sizes, pad_mode='symmetric', constant_values=0, combine=0):

    # check data type
    is_list = isinstance(array_list, list)
    is_tuple = isinstance(array_list, tuple)
    if not (is_list or is_tuple):
        array_list = [array_list]
    if is_tuple:
        array_list = list(array_list)

    if isinstance(pad_sizes, int):
        pad_sizes = [pad_sizes]

    if isinstance(constant_values, int):
        constant_values = [constant_values]

    n_items = len(array_list)
    n_dims = len(array_list[0].shape)

    # get pad dimensions
    pad_dims = []
    pad_length = len(pad_sizes)
    if pad_length == 1:
        for j in range(n_dims):
            pad_dims.append((pad_sizes, pad_sizes))
    elif pad_length == n_dims:
        for j in range(n_dims):
            pad_dims.append((pad_sizes[j], pad_sizes[j]))
    elif pad_length == n_dims*2:
        pad_dims = pad_sizes
    else:
        print('Wrong pad sizes')


    # # pad values
    # pad_vals = []
    # val_length = len(constant_values)
    # if val_length == 1:
    #     for j in range(n_dims):
    #         pad_vals.append((constant_values, constant_values))
    # elif val_length == n_dims:
    #     for j in range(n_dims):
    #         pad_vals.append((constant_values[j], constant_values[j]))
    # elif val_length == n_dims*2:
    #     pad_vals = constant_values
    # else:
    #     print('Wrong pad values')
    # print(pad_val

    # apply padding
    for i in range(n_items):
        item = array_list[i]
        is_tensor = isinstance(item, torch.Tensor)
        if is_tensor:
            item = item.numpy()

        if pad_mode != 'constant':
            item = np.pad(item, pad_dims, pad_mode)
        else:
            item = np.pad(item, pad_dims, pad_mode, constant_values=constant_values)

        if is_tensor:
            item = torch.tensor(item)

        array_list[i] = item


    # convert to return    
    if not (is_list or is_tuple):
        return array_list[0]
    else:
        if combine == 0 and is_tuple:
            array_list = tuple(array_list)
        elif combine == 1:
            array_list = np.concatenate(array_list, axis=0)
            # print('cat')
        elif combine == 2:
            array_list = np.vstack(array_list)
            # print('stack')


    return array_list
