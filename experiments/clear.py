import glob
import os

result_fps = glob.glob('./results/*.yml')

for result_fp in result_fps:
    if os.stat(result_fp).st_size == 0:
        print(f'Removing {result_fp}...')
        os.remove(result_fp)
