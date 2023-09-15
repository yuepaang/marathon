import time
import rust_perf

if __name__ == "__main__":
    start = (0, 0)
    end = (2, 3)
    blocks = [(1, 0), (1, 1), (1, 2), (2, 2)]
    s = time.time()
    print("start: ", start)
    print("end: ", end)
    print("blocks: ", blocks)
    print(rust_perf.get_direction(start, end, blocks))
    print(rust_perf.get_direction_path(start, end, blocks))
    print(time.time() - s)
