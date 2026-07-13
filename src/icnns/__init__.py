from typing import Callable, Iterable
import jax
import jaxtyping
from jax import lax
import jax.numpy as jnp
import equinox as eqx


class Linear(eqx.Module):
    """単層FICNN"""

    weight_z: jax.Array
    weight_y: jax.Array | None
    bias: jax.Array | None
    input_size: int = eqx.field(static=True)
    linear_input_size: int | None = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    with_linear: bool = eqx.field(static=True)
    with_bias: bool = eqx.field(static=True)
    transform: Callable[[jax.Array], jax.Array] = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        linear_input_size: int | None,
        output_size: int,
        with_linear: bool = True,
        with_bias: bool = True,
        transform: Callable[[jax.Array], jax.Array] = jax.nn.softplus,
        *,
        key=jaxtyping.PRNGKeyArray,
    ):
        """
        Args:
        input_size: 入力テンソルの次元
        linear_input_size: 線形入力テンソルの次元
        output_size: 出力テンソルの次元
        with_linear: 線形項を加えるかどうか
        with_bias: バイアスを加えるかどうか
        transform: 重みを正の数に変換するための関数。 Default: jax.nn.softplus.
        key: 乱数キー
        """
        self.input_size = input_size
        self.linear_input_size = linear_input_size
        self.output_size = output_size
        self.transform = transform
        self.with_linear = with_linear
        self.with_bias = with_bias

        key1, key2, key3 = jax.random.split(key, 3)

        fan = input_size + (linear_input_size if linear_input_size is not None else 0)
        stddev = jnp.sqrt(2.0 / fan)
        mean = jnp.sqrt(2.0 / jnp.pi) * stddev
        sigma = jnp.sqrt(stddev**2 - mean**2)
        lower = -mean / sigma
        upper = jnp.inf
        self.weight_z = (
            jax.random.truncated_normal(
                key1, lower=lower, upper=upper, shape=(input_size, output_size)
            )
            * sigma
        )

        if with_linear:
            if linear_input_size is None:
                raise ValueError(
                    "linear_input_size must be provided when with_linear is True."
                )
            self.weight_y = (
                jax.random.normal(key2, shape=(linear_input_size, output_size)) * stddev
            )

            if with_bias:
                self.bias = jax.random.normal(key3, shape=(output_size,))
            else:
                self.bias = None
        else:
            self.weight_y = None
            self.bias = None

    def __call__(
        self,
        z: jax.Array,
        y: jax.Array | None = None,
        *,
        precision: lax.Precision | None = None,
    ) -> jax.Array:
        """
        Computes a linear transform
            u(z, x) = Wz z + Wy y + b.
        z はyについて凸な関数の出力。
        Wz はzにかける正の重み。
        Wy はyにかける重み。
        b はバイアス。
        with_linearがFalseのとき、Wy = 0, b = 0になる。
        with_biasがFalseのとき、b = 0になる。

        Args:
        z: 凸関数の出力テンソル。(*batch_size, input_size)の形状。
        y: 入力テンソル。(*batch_size, linear_input_size)の形状。with_linearがTrueのときに必要。
        """

        out = jnp.dot(z, self.transform(self.weight_z), precision=precision)

        if self.with_linear:
            out = out + jnp.dot(y, self.weight_y, precision=precision)

            if self.with_bias:
                out = out + self.bias

        return out
