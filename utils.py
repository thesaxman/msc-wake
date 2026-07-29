from contextlib import contextmanager
import time

@contextmanager
def timed(label):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"{label}: {time.perf_counter() - t0:.3f} s")
