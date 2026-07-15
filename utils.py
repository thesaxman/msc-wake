from contextlib import contextmanager

@contextmanager
def timed(label):
    t0 = time.perf_counter()
    yield
    print(f"{label}: {time.perf_counter() - t0:.3f} s")
