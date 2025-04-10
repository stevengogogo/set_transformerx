import equinox as eqx
import jax
from jax import random as jr
from .modules import ISAB, SAB, PMA

class SetTransformer(eqx.Module):
    enc: eqx.Module
    dec: eqx.Module
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False, key=jr.PRNGKey(0)):
        ks = jr.split(key, 6)
        self.enc = eqx.nn.Sequential([
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln, key=ks[0]),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln, key=ks[1])])
        self.dec = eqx.nn.Sequential([
                PMA(dim_hidden, num_heads, num_outputs, ln=ln, key=ks[2]),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln, key=ks[3]),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln, key=ks[4]),
                jax.vmap(eqx.nn.Linear(dim_hidden, dim_output, key=ks[5]))])

    def __call__(self, X):
        X = self.enc(X)
        X = self.dec(X)
        return X