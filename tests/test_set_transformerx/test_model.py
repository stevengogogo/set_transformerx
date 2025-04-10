
from set_transformerx.modules import MAB, SAB, ISAB
from set_transformerx.models import SetTransformer
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

def test_tf():
    n = 10
    dim_input = 11
    dim_output = 12
    X = jnp.ones((n, dim_input))

    tf = SetTransformer(dim_input=dim_input, num_outputs=4, dim_output=dim_output, num_inds=2, dim_hidden=12, num_heads=2, ln=True)
    O = tf(X)

    assert O.shape == (n, dim_output)