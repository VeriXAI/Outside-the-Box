def uniform_partition(n, block_size):
    m = n // block_size
    r = n % block_size
    partition = [block_size for _ in range(m)]
    if r > 0:
        partition.append(r)
    return partition
