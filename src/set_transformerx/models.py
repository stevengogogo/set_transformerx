import equinox as eqx
import jax
from jax import random as jr
from .modules import ISAB, SAB, PMA

class SetTransformer(eqx.Module):
    enc: eqx.Module
    dec: eqx.Module
    out: eqx.Module
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False, *, mlp_kwargs=None, key=jr.PRNGKey(0)):
        ks = jr.split(key, 6)
        self.enc = [
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln, mlp_kwargs=mlp_kwargs, key=ks[0]),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln, mlp_kwargs=mlp_kwargs, key=ks[1])]
        self.dec = [
                PMA(dim_hidden, num_heads, num_outputs, ln=ln, mlp_kwargs=mlp_kwargs, key=ks[2]),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln, mlp_kwargs=mlp_kwargs, key=ks[3]),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln, mlp_kwargs=mlp_kwargs, key=ks[4])]
        
        self.out = eqx.nn.Linear(dim_hidden, dim_output, key=ks[5])

    def __call__(self, X):
        for b in self.enc:
                X = b(X)
        for b in self.dnc:
                X = b(X)
        X = self.dec(X)
        X = jax.vmap(self.out)(X)
        return X

