import numpy as np

def positional_encoding(x, L=10):
    """
    Applies NeRF's positional encoding to a given input coordinate x.

    Args:
    - x: Input tensor (N, D) where N is the number of points and D is the input dimension.
    - L: Number of frequency bands.

    Returns:
    - Encoded tensor (N, 2*L*D)
    """
    encoded = [x]  # Start with raw input
    for i in range(L):
        encoded.append(np.sin(2.0**i * np.pi * x))
        encoded.append(np.cos(2.0**i * np.pi * x))
    return np.concatenate(encoded, axis=-1)

# Example usage
x = np.array([[0.1, 0.2, 0.3]])  # Example 3D point
encoded_x = positional_encoding(x, L=10)
print(encoded_x.shape)  # Output: (1, 63), since 3D -> (1 + 2*10) * 3 = 63

