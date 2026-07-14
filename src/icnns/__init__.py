from typing import Callable
import jax
import jaxtyping
from jax import lax
import jax.numpy as jnp
import equinox as eqx
import sys


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


class ConvexLinear(eqx.Module):
    """単層ICNN"""

    weight_z: jax.Array | None
    weight_y: jax.Array | None
    bias: jax.Array | None
    convex_input_size: int = eqx.field(static=True)
    linear_input_size: int | None = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    with_convex_funcs: bool = eqx.field(static=True)
    with_linear: bool = eqx.field(static=True)
    with_bias: bool = eqx.field(static=True)
    transform: Callable[[jax.Array], jax.Array] = eqx.field(static=True)

    def __init__(
        self,
        convex_function_size: int,
        input_size: int,
        output_size: int,
        with_bias: bool = True,
        transform: Callable[[jax.Array], jax.Array] = jnp.exp,
        *,
        key=jaxtyping.PRNGKeyArray,
    ):
        """
        Args:
        convex_function_size: 凸関数入力テンソルの次元
        input_size: 線形入力テンソルの次元
        output_size: 出力テンソルの次元
        with_bias: バイアスを加えるかどうか
        transform: 重みを正の数に変換するための関数。 Default: jax.nn.softplus.
        key: 乱数キー
        """
        self.convex_input_size = convex_function_size
        self.linear_input_size = input_size
        self.output_size = output_size
        self.transform = transform
        self.with_convex_funcs = convex_function_size > 0
        self.with_linear = input_size > 0
        self.with_bias = with_bias

        key1, key2, key3 = jax.random.split(key, 3)

        if self.with_convex_funcs:
            fan_conv = self.convex_input_size
            stddev = jnp.sqrt(2.0 / fan_conv)
            # (e^(sigma_v^2) - 1) * e^(sigma_v^2) = stddev^2
            t = (1.0 + jnp.sqrt(1.0 + 4.0 * stddev**2)) * 0.5
            sigma_v = jnp.sqrt(jnp.log(t))
            self.weight_z = (
                jax.random.normal(
                    key1, shape=(self.convex_input_size, self.output_size)
                )
                * sigma_v
            )

            mu_w = jnp.exp((sigma_v**2) * 0.5)
            b_val = -fan_conv * mu_w
        else:
            self.weight_z = None
            b_val = 0.0

        if self.with_linear:
            fan = self.linear_input_size
            stddev = jnp.sqrt(2.0 / fan)
            self.weight_y = (
                jax.random.normal(
                    key2, shape=(self.linear_input_size, self.output_size)
                )
                * stddev
            )

        else:
            self.weight_y = None

        if with_bias:
            self.bias = b_val + jax.random.normal(key3, shape=(self.output_size,))
        else:
            self.bias = None

    def __call__(
        self,
        z: jax.Array | None,
        y: jax.Array | None,
        *,
        precision: lax.Precision | None = None,
    ) -> jax.Array:
        """
        Computes a linear transform
            u(z, x) = Wz z + Wy y + b.
        z : yについて凸な関数の出力
        Wz: zにかける正の重み
        Wy: yにかける重み
        b : バイアス
        with_biasがFalseのとき、b = 0になる。

        Args:
        z: 凸関数の入力テンソル。(*batch_size, convex_functions)の形状。
        y: 入力テンソル。(*batch_size, linear_input_size)の形状。with_linearがTrueのときに必要。
        """

        if self.weight_z is not None:
            if z is not None:
                out1 = jnp.dot(z, self.transform(self.weight_z), precision=precision)
            else:
                raise ValueError("z must be provided when convex_function_size > 0.")
        else:
            out1 = jnp.array(0.0)
            if z is not None:
                print(
                    "z is provided but convex_function_size = 0, ignoring z.",
                    file=sys.stderr,
                )

        if self.weight_y is not None:
            if y is not None:
                out2 = jnp.dot(y, self.weight_y, precision=precision)
            else:
                raise ValueError("y must be provided when linear_input_size > 0.")
        else:
            out2 = jnp.array(0.0)
            if y is not None:
                print(
                    "y is provided but linear_input_size = 0, ignoring y.",
                    file=sys.stderr,
                )

        if self.bias is not None:
            out3 = self.bias
        else:
            out3 = jnp.array(0.0)

        return out1 + out2 + out3
