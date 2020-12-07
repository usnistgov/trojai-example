from multiprocessing import Pool
import time
import os


def f(x):
    print(x*x)
    with open(str(os.getpid())+'.out','w') as f:
        f.write(str(os.getpid()) + str(x*x))
    return None

if __name__=='__main__':
    with Pool(processes=4) as pool:
        for i in range(10):
            pool.apply(f,(i,))
    time.sleep(10)

