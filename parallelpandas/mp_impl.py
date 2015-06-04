import multiprocessing as mp
import pandas as pd
import dill
from functools import wraps

__spawn_ctx__ = None


def get_mp_ctx(f):
    def with_ctx(*args, **kws):
        global __spawn_ctx__
        if __spawn_ctx__ is None:
            __spawn_ctx__ = mp.get_context('spawn')
        return f(*args, mp_ctx=__spawn_ctx__, **kws)
    return wraps(f)(with_ctx)


def apply_worker(result_queue, data, offset, chunk_size, f_pickled, kwargs):
    # Unpickle the function
    f = dill.loads(f_pickled)
    # Apply function and store result
    chunk = data[offset*chunk_size:(offset+1)*chunk_size]
    result_queue.put((offset, chunk.apply(f, **kwargs)))
    return True


@get_mp_ctx
def apply(data, f, n_processes=mp.cpu_count(), mp_ctx=None, **kwargs):
    """
    parallel version of apply

    Parameters
    ----------
    data : Series or DataFrame
        the pandas object to apply data on
    f : function
        the function to apply
    n_processes : int (optional)
        specify the number of processes to use, otherwise use all
    Additional keyword arguments will be passed on to pandas apply operation

    Returns
    -------
    result of apply operation
    """
    # Compute chunk size
    chunk_size = (len(data)-1) // n_processes + 1
    # Pickle function
    f_pickled = dill.dumps(f)
    # Start workers
    result_queue = mp.Queue()
    processes = []
    for offset in range(n_processes):
        p = mp.Process(target=apply_worker,
                       args=(result_queue, data, offset, chunk_size,
                             f_pickled, kwargs))
        p.daemon = True
        p.start()
        processes.append(p)
    # Get results
    results = list()
    while len(results) < n_processes:
        results.append(result_queue.get())
    # Wait for all workers to finish
    for p in processes:
        p.join()
    # Close queues
    result_queue.close()
    # Sort results such that resulting series/dataframe has the same index
    # as the original one
    results = [y[1] for y in sorted(results, key=lambda x: x[0])]
    # Return concated results
    return pd.concat(results)


def groupby_apply_worker(result_queue, dataframe, columns, f_pickled,
                         i_segment, segments):
    # Unpickle the function
    f = dill.loads(f_pickled)
    # Groupby, apply function and return result
    result_queue.put(dataframe[segments == i_segment].groupby(
        by=columns, sort=False).apply(f))
    return True


@get_mp_ctx
def groupby_apply(dataframe, columns, func,
                  n_processes=mp.cpu_count(), mp_ctx=None):
    """
    parallel version of groupby/apply

    Parameters
    ----------
    data : Series or DataFrame
        the pandas object to apply data on
    columns : list or Index
        subset of columns to used to group by
    func : function
        the function pass to apply of the grouper object
    n_processes : int (optional)
        specify the number of processes to use, otherwise use all

    Returns
    -------
    result of groupby/apply operation
    """
    # Pickle function
    f_pickled = dill.dumps(func)
    # Segmentize rows such that two rows belonging to the same group
    # are assigned to the same segment
    segments = apply(dataframe[columns],
                     lambda x: hash(tuple(x.values)) % n_processes,
                     n_processes=mp.cpu_count(), axis=1)
    # Init and start workers
    result_queue = mp.Queue()
    processes = []
    for iSegment in range(n_processes):
        p = mp.Process(target=groupby_apply_worker,
                       args=(result_queue, dataframe, columns,
                             f_pickled, iSegment, segments))
        p.daemon = True
        p.start()
        processes.append(p)
    # Get results
    results = list()
    while len(results) < n_processes:
        results.append(result_queue.get())
    # Wait for all workers to finish
    for p in processes:
        p.join()
    # Close queues
    result_queue.close()
    # Return concated results
    return pd.concat(results)
