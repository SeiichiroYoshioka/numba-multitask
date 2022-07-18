import numpy as np
from tqdm import tqdm
from time import perf_counter
from numba import jit
from concurrent.futures import ThreadPoolExecutor


def notuse_numpyapi(A, B):
    result = np.zeros(shape=(len(A), len(B)))
    for i in tqdm(range(len(A))):
        for j in range(len(B)):
            result[i, j] = np.dot(A[i], B[j])
    return result


@jit("float64[:, :](float64[:,:], float64[:, :])", nopython=True, nogil=True)
def use_numpyapi_nogil(A, B):
    result = np.zeros(shape=(len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            result[i, j] = np.dot(A[i], B[j])
    return result


@jit("float64[:, :](float64[:,:], float64[:, :])", nopython=True)
def use_numpyapi(A, B):
    result = np.zeros(shape=(len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            result[i, j] = np.dot(A[i], B[j])
    return result


def main():
    randobj = np.random.RandomState(100)

    A = randobj.random(size=(1000, 10))
    B = randobj.random(size=(2000, 10))
    C = randobj.random(size=(3000, 10))

    start_notuse = perf_counter()
    x = notuse_numpyapi(A, B)
    y = notuse_numpyapi(A, C)
    end_notuse = perf_counter()

    start_use = perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(use_numpyapi, A, B)
        future2 = executor.submit(use_numpyapi, A, C)
    x = future1.result()
    y = future2.result()
    end_use = perf_counter()

    start_use_nogil = perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(use_numpyapi_nogil, A, B)
        future2 = executor.submit(use_numpyapi_nogil, A, C)
    x = future1.result()
    y = future2.result()
    end_use_nogil = perf_counter()

    print(f"Not use numpy api time {end_notuse - start_notuse}")
    print(f"Use numpy api time {end_use - start_use}")
    print(f"Use numpy api time {end_use_nogil - start_use_nogil}")
    return


if __name__ == "__main__":
    main()
