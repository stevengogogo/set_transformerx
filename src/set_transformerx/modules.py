"""
Referece: https://github.com/stevengogogo/set_transformer/blob/master/modules.py
"""
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random as jr
from typing import Callable, Optional

class MAB(eqx.Module):
    fc_q: eqx.nn.Linear 
    fc_k: eqx.nn.Linear
    fc_qv: eqx.nn.Linear
    fc_o: eqx.nn.Linear
    res1: Callable # residual connection 1
    res2: Callable # residual connection 2
    dim_V: int 
    num_heads: int

    def __init__(self, dim_Q, dim_K, dim_V, num_heads:int, *, ln=False, mlp_kwargs: Optional[dict]=None, key=jr.PRNGKey(0), **kwargs):
        """
        ln: layernorm
        """
        k = jr.split(key, 4)
        self.dim_V = dim_V 
        self.num_heads = num_heads 
        self.fc_q = eqx.nn.Linear(dim_Q, dim_V, key=k[0])
        self.fc_k = eqx.nn.Linear(dim_K, dim_V, key=k[1])
        self.fc_qv = eqx.nn.Linear(dim_K, dim_V, key=k[2])

        if ln: 
            self.res1 = eqx.nn.LayerNorm(dim_V)
            self.res2 = eqx.nn.LayerNorm(dim_V)
        else:
            self.res1 = lambda x: x
            self.res2 = lambda x: x

        if mlp_kwargs is None:
            mlp_kwargs = dict(
                width_size=None,
                final_activation=jax.nn.relu,
                depth=0
            )
        self.fc_o = eqx.nn.MLP(in_size=dim_V, out_size=dim_V, **mlp_kwargs,key=k[3])

    def __call__(self, Q, K, **kwargs):
        q = jax.vmap(self.fc_q)(Q)
        k = jax.vmap(self.fc_k)(K)
        v = jax.vmap(self.fc_qv)(K)
        q = jnp.expand_dims(q, axis=0)
        k = jnp.expand_dims(k, axis=0)
        v = jnp.expand_dims(v, axis=0)


        dim_split = self.dim_V // self.num_heads
        Q_ = jnp.concatenate(jnp.split(q, dim_split, axis=2), axis=0)
        K_ = jnp.concatenate(jnp.split(k, dim_split, axis=2), axis=0)
        V_ = jnp.concatenate(jnp.split(v, dim_split, axis=2), axis=0)

        attn_logits = jax.lax.batch_matmul(Q_, K_.transpose(0,2,1)) / jnp.sqrt(self.dim_V)
        
        A = jax.nn.softmax(attn_logits)
        
        O = (Q_ + jax.lax.batch_matmul(A, V_)).reshape(-1,self.dim_V)
        
        # Residual connection
        O = jax.vmap(self.res1)(O)
        O = O + jax.vmap(self.fc_o)(O)
        O = jax.vmap(self.res2)(O)
        return O

class SAB(eqx.Module):
    mab: MAB
    def __init__(self, dim_in, dim_out, num_heads, ln=False, *, mlp_kwargs:Optional[dict]=None, key=jr.PRNGKey(0), **kwargs):
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln, key=key, mlp_kwargs=mlp_kwargs, **kwargs)
    def __call__(self, X, **kwargs):
        return self.mab(X,X, **kwargs)

class ISAB(eqx.Module):
    I: jnp.ndarray
    mab0: MAB
    mab1: MAB
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False,*, mlp_kwargs:Optional[dict]=None, key=jr.PRNGKey(0), **kwargs):
        ks = jr.split(key, 2)
        init = jax.nn.initializers.glorot_uniform()
        self.I = init(key, (num_inds, dim_out)) 
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln, key=ks[0], mlp_kwargs=mlp_kwargs, **kwargs) 
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln, key=ks[1], mlp_kwargs=mlp_kwargs, **kwargs)
    def __call__(self, X, **kwargs):
        H = self.mab0(self.I,X, **kwargs)
        return self.mab1(X,H, **kwargs)

class PMA(eqx.Module):
    S: jnp.ndarray
    mab: MAB
    enc: eqx.Module
    def __init__(self, dim, num_heads, num_seeds, ln=False, *, mlp_kwargs:Optional[dict]=None, key=jr.PRNGKey(0), **kwargs):
        ks = jr.split(key, 3)
        init = jax.nn.initializers.glorot_uniform()
        self.S = init(ks[0], (num_seeds, dim))
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln, mlp_kwargs=mlp_kwargs, key=ks[1], **kwargs)
        self.enc = eqx.nn.MLP(in_size=dim, out_size=dim, **mlp_kwargs, key=ks[2])

    def __call__(self, X, **kwargs):
        H = self.mab(self.S,X, **kwargs)
        return H