import numpy as np

def sample_parity(K, parity=0):
    """
    Sample x in {-1,1}^K *uniformly* subject to 
    prod_i x[i] = (-1)**parity.
    
    Parameters
    ----------
    K : int
        Length of vector.
    parity : {0,1}
        0 ⇒ even number of -1’s ⇒ prod = +1
        1 ⇒ odd number of -1’s  ⇒ prod = -1
    
    Returns
    -------
    x : np.ndarray of shape (K,), dtype=int
        A random vector satisfying the parity constraint.
    """
    # sample first K-1 entries freely
    x = np.random.choice([-1,1], size=K)
    # compute their product
    p = x[:-1].prod()
    # pick the last entry so that overall product = (-1)**parity
    xK = ((-1)**parity) // p
    x[-1] = xK
    # concatenate and return
    return x


def sample_parity_vec(N, K, parity=0):
    x = np.random.choice([-1,1], size=(N, K))
    pk_1 = x[:, :-1].prod(axis=1)
    xK = ((-1)**parity) // pk_1
    x[:, -1] = xK
    return x


def sample_group_parity_vec(N, sample_len, group_size=1, parity=0):
    num_groups = sample_len // group_size
    x = sample_parity_vec(N=N * num_groups, K=group_size, parity=parity)
    x = x.reshape(N, num_groups * group_size)
    return x


def parity_func(x, axis=0):
    parity = x.prod(axis=axis) # this is -1, 1 -1 is odd, mapping to 1, 1 is even mapping to 0
    return (- parity + 1) // 2



def sample_ensuring_uniqueness(N, sample_func):
    collected_samples = []
    seen = set()
    
    while len(collected_samples) < N:
        # Sample more than needed to reduce iterations
        batch_size = max(N - len(collected_samples), N // 2)
        x_batch = sample_func(N=batch_size)
        # Convert batch to tuples for set operations
        batch_tuples = set(tuple(row) for row in x_batch)
        # Find new unique samples
        new_samples = batch_tuples - seen
        # Update seen set
        seen.update(new_samples)
        # Add new samples to collected samples
        for sample_tuple in new_samples:
            collected_samples.append(np.array(sample_tuple))
            if len(collected_samples) == N:
                break
    
    return np.array(collected_samples)


if __name__ == "__main__":
    # Example:
    x = sample_parity(5, parity=1)
    print(x, x.prod())
    x = sample_parity_vec(N=10, K=6, parity=0)
    print(x, parity_func(x, axis=1))
    x = sample_group_parity_vec(N=4096, sample_len=64, group_size=8, parity=0)
    num_unique = len(np.unique(x, axis=0))
    print(num_unique)

    sample_len = 64
    for group_size in [2, 4, 8, 16, 32, 64]:
        x = sample_ensuring_uniqueness(N=4096, sample_func=lambda N: sample_group_parity_vec(N=N, sample_len=sample_len, group_size=group_size, parity=0))
        num_unique = len(np.unique(x, axis=0))
        print(f"Even parity, group_size: {group_size}, num_unique: {num_unique}")
        x = sample_ensuring_uniqueness(N=4096, sample_func=lambda N: sample_group_parity_vec(N=N, sample_len=sample_len, group_size=group_size, parity=1))
        num_unique = len(np.unique(x, axis=0))
        print(f"Odd parity,  group_size: {group_size}, num_unique: {num_unique}")