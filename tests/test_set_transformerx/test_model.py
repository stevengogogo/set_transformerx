
from set_transformerx.model import MAB, SAB, ISAB
import jax.numpy as jnp

def test_mab():
    # test the MAB
    n = 10
    dim_Q = 11
    dim_K =  dim_V =12
    num_heads = 2

    Q = jnp.ones((n, dim_Q))
    K = jnp.ones((n, dim_V))
    mab = MAB(dim_Q, dim_K, dim_V, num_heads, ln=False)
    sab = SAB(dim_Q, dim_V, num_heads, ln=False)
    isab = ISAB(dim_Q, dim_V, num_heads, 5, ln=False)

    O = mab(Q, K)
    O1 = sab(Q)
    O2 = isab(Q)


    assert O.shape == (n, dim_V)
    assert O1.shape == O2.shape
