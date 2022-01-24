from multiprocessing import Pool

def pool(worker, worker_list, multi=True, cp=None):
    """
    Simple wrapper to force pool to be synchronous (for memory purposes)
    """
    if multi:
        if cp is not None: cp("%s Pools" % len(worker_list))
        with Pool() as pl:
            return pl.map(worker, worker_list)
    else:
        if cp is not None: cp("%s sync list" % len(worker_list))
        return [worker(i) for i in worker_list]