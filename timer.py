from functools import wraps
import time

def timer(pre_str=None):
    def timed(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            start_time = time.time()
            res = func(*args, **kwargs)
            print_srt = pre_str or func.__name__
            print(f"| {print_srt} cost {time.time() - start_time:.3f} seconds.")
            return res
        return wrapped
    return timed