import torch as t
from torch import nn
import torch.nn.functional as F
import math
import numpy as np

class AddAndNorm(nn.Module):
    def __init__(self, hidden_layer_size):
        super(AddAndNorm, self).__init__()

        self.normalize = nn.LayerNorm(hidden_layer_size)

    def forward(self, x1, x2, x3=None):
        if x3 is not None:
            x = t.add(t.add(x1, x2), x3)
        else:
            x = t.add(x1, x2)
        return self.normalize(x)

class GLU(nn.Module):
    # Gated Linear Unit+
    def __init__(self,
                 input_size,
                 hidden_layer_size,
                 dropout_rate=None,
                 ):
        super(GLU, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)
        self.activation_layer = t.nn.Linear(input_size, hidden_layer_size)
        self.gated_layer = t.nn.Linear(input_size, hidden_layer_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.dropout_rate is not None:
            x = self.dropout(x)
        activation = self.activation_layer(x)
        gated = self.sigmoid(self.gated_layer(x))
        return t.mul(activation, gated), gated

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = t.zeros(max_len, d_model)
        position = t.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0).unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :, :x.size(2), :]



def calculate_ttc(pos1, vel1, pos2, vel2, radius1, radius2):
    """
    Calculate Time to Collision (TTC) for two objects in 2D space.

    Parameters:
    pos1: numpy array, position of object 1 [x, y]
    vel1: numpy array, velocity of object 1 [vx, vy]
    pos2: numpy array, position of object 2 [x, y]
    vel2: numpy array, velocity of object 2 [vx, vy]
    radius1: float, radius of object 1
    radius2: float, radius of object 2

    Returns:
    float: Time to collision (TTC) in seconds, or float('inf') if no collision
    """
    # Relative position and velocity
    rel_pos = pos2 - pos1
    rel_vel = vel2 - vel1

    # Combined radius
    r = radius1 + radius2

    # Coefficients for quadratic equation: a*t^2 + b*t + c = 0
    a = np.sum(rel_vel * rel_vel, axis=1)  # dot product of relative velocity
    b = 2 * np.sum(rel_pos * rel_vel, axis=1)  # 2 * dot product of relative position and velocity
    c = np.sum(rel_pos * rel_pos, axis=1) - r ** 2  # adjusted for radii

    # Initialize TTC array
    ttc = np.full(a.shape, np.inf)

    # Avoid division by zero where relative velocity is zero
    non_zero_a = np.abs(a) > 1e-10

    # Calculate discriminant
    discriminant = b[non_zero_a] ** 2 - 4 * a[non_zero_a] * c[non_zero_a]

    # Only consider cases with real roots (discriminant >= 0)
    valid = discriminant >= 0
    valid_indices = np.where(non_zero_a)[0][valid]

    # Solve for positive roots
    sqrt_disc = np.sqrt(discriminant[valid])
    a_valid = a[valid_indices]
    b_valid = b[valid_indices]

    # Calculate possible collision times
    t1 = (-b_valid + sqrt_disc) / (2 * a_valid)
    t2 = (-b_valid - sqrt_disc) / (2 * a_valid)

    # Select the smallest positive time
    for i, idx in enumerate(valid_indices):
        times = [t for t in [t1[i], t2[i]] if t > 0]
        if times:
            ttc[idx] = min(times)

    return ttc


if __name__ == "__main__":
    p1 = np.array([[0, 0],[0, 0]])
    v1 = np.array([[0, 1],[0, 1]])
    p2 = np.array([[0, 5], [20, 5]])
    v2 = np.array([[0, 0], [0, 0]])
    r1=np.array([1,1])
    r2 = np.array([2, 3])
    #
    # p1 = np.array([0, 0])
    # v1 = np.array([0, 1])
    # p2 = np.array([0, 5])
    # v2 = np.array([0, 0])
    # r1 = 1
    # r2 = 2

    ttc1 = calculate_ttc(p1, v1, p2, v2, r1, r2)
    print(f"TTC = {ttc1:.3f} s")  