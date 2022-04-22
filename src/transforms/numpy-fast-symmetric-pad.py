"""
MIT License

Copyright (c) 2017 Stian Lode,
                   stian.lode@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import numpy as np

# Let me know if you found it useful at stian.lode@gmail.com
def edges(window):
    """Splits a window into start and end indices. Will default to have the
    larger padding at the end in case of an odd window.
    """
    start = window//2
    end = window-start
    return (start, end)

def fast_pad_symmetric(values, window, dtype='f8'):
    """A fast version of numpy n-dimensional symmetric pad.

    In contrast to np.pad, this algorithm only allocates memory once, regardless
    of the number of axes padded. Performance for large data sets is vastly
    improved.

    Note: if the requested padding is 0 along all axes, then this algorithm
    returns the original input ndarray.

    Author: Stian Lode stian.lode@gmail.com

    Args:
        values: n-dimensional ndarray
        window: an iterable of length n

    return:
        a numpy ndarray containing the values with each axis padded according
        to the specified window. The padding is a reflection of the data in 
        the input values.
    """
    assert len(values.shape) == len(window) 

    if (window <= 0).all():
        return values

    start, end = edges(window)
    new = np.empty(values.shape + window, dtype=dtype)

    slice_stack = []
    for a, b in zip(start, end):
        slice_stack.append(slice(a, None if b == 0 else -b))

    new[tuple(slice_stack)] = values

    slice_stack = []
    for a,b in zip(start, end):
        if a > 0:
            s_to, s_from = slice(a - 1, None, -1), slice(a, 2 * a, None)
            new[tuple(slice_stack + [s_to])] = new[tuple(slice_stack + [s_from])]

        if b > 0:
            e_to, e_from = slice(-1, -b-1, -1), slice(-2 * b, -b)
            new[tuple(slice_stack + [e_to])] = new[tuple(slice_stack + [e_from])]

        slice_stack.append(slice(None))

    return new

def numpypad(values, window):
    s, e = edges(window)
    return np.pad(values, list(zip(s, e)), mode='symmetric')

import time
for N in [100, 200, 300, 400, 500, 600, 800]:
    values = np.arange(N*N*N, dtype=np.float).reshape(N,N,N)
    window = np.array((8,8,8), dtype=np.int)

    to = time.clock()
    a = numpypad(values, window)
    print("numpypad {} {}".format(N, time.clock()-to))

    to = time.clock()
    b = fast_pad_symmetric(values, window)
    print("fast_pad_symmetric {} {}".format(N, time.clock()-to))

    import gc; gc.collect()
    assert (a==b).all()