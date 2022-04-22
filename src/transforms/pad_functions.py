# %%
import numpy as np

import torch



# %%
def collate_fn_padd(batch, device=None):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    lengths = torch.tensor([ t.shape[0] for t in batch ]).to(device)
    ## padd
    batch = [ torch.Tensor(t).to(device) for t in batch ]
    batch = torch.nn.utils.rnn.pad_sequence(batch)
    ## compute mask
    mask = (batch != 0).to(device)
    return batch, lengths, mask


def collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    _, labels, lengths = zip(*data)
    max_len = max(lengths)
    n_ftrs = data[0][0].size(1)
    features = torch.zeros((len(data), max_len, n_ftrs))
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)

    for i in range(len(data)):
        j, k = data[i][0].size(0), data[i][0].size(1)
        features[i] = torch.cat([data[i][0], torch.zeros((max_len - j, k))])

    return features.float(), labels.long(), lengths.long()


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