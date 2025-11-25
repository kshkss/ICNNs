from typing import Callable, Iterable
import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


class ConvexLinear(hk.Module):
    """単層FICNN"""

    def __init__(
        self,
        output_size: int,
        with_linear: bool = True,
        with_bias: bool = True,
        transform: Callable[[jax.Array], jax.Array] = jax.nn.softplus,
        w_init: hk.initializers.Initializer | None = None,
        b_init: hk.initializers.Initializer | None = None,
        name: str | None = None,
    ):
        """
        Args:
        output_size: 出力テンソルの次元
        with_linear: 線形項を加えるかどうか
        with_bias: バイアスを加えるかどうか
        transform: 重みを正の数に変換するための関数。 Default: jax.nn.softplus.
        name: Name of the module.
        """
        super().__init__(name=name)
        self.input_size = None
        self.linear_input_size = None
        self.output_size = output_size
        self.transform = transform
        self.with_linear = with_linear
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros

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
        if not z.shape:
            raise ValueError("Input must not be scalar.")

        self.input_size = z.shape[-1]
        dtype = z.dtype

        w_init = self.w_init
        if w_init is None:
            stddev = 1.0 / np.sqrt(self.input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        wz = hk.get_parameter(
            "wz", [self.input_size, self.output_size], dtype, init=w_init
        )

        out = jnp.dot(z, self.transform(wz), precision=precision)

        if self.with_linear:
            self.linear_input_size = y.shape[-1]
            if w_init is None:
                stddev = 1.0 / np.sqrt(self.linear_input_size)
                w_init = hk.initializers.TruncatedNormal(stddev=stddev)
            wy = hk.get_parameter(
                "wy", [self.linear_input_size, self.output_size], dtype, init=w_init
            )
            out = out + jnp.dot(y, wy, precision=precision)

            if self.with_bias:
                b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
                b = jnp.broadcast_to(b, out.shape)
                out = out + b

        return out


class FICNN(hk.Module):
    """A fully input convex neural network."""

    def __init__(
        self,
        output_sizes: Iterable[int],
        w_init: hk.initializers.Initializer | None = None,
        b_init: hk.initializers.Initializer | None = None,
        with_bias: bool = True,
        activation: Callable[[jax.Array], jax.Array] = jax.nn.softplus,
        activate_final: bool = False,
        linear_final: bool = False,
        bias_final: bool = False,
        convex_input: bool = False,
        transform: Callable[[jax.Array], jax.Array] = jax.nn.softplus,
        name: str | None = None,
    ):
        """
        Args:
        output_sizes: 各層の出力の次元
        w_init: Initializer for weights.
        b_init: Initializer for bias. Must be None if with_bias=False.
        with_bias: 全層でバイアスを加えるかどうか
        activation: 活性化関数。下に凸で単調増加な関数であること。 Default: jax.nn.softplus.
        activate_final: 最後の層の出力に活性化関数を適用するかどうか。
        linear_final: 最後の層に線形項を加えるかどうか。
        bias_final: 最後の層にバイアスを加えるかどうか。
        convex_input: 入力yに追加で凸関数の出力zを入力に受け取るかどうか。
        transform: 重みを正の数に変換するための関数。 Default: jax.nn.softplus.
        name: Optional name for this module.
        """
        super().__init__(name=name)
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init
        self.activation = activation
        self.activate_final = activate_final
        self.linear_final = linear_final
        self.bias_final = bias_final
        self.convex_input = convex_input
        self.transform = transform
        layers = []
        output_sizes = tuple(output_sizes)
        for index, output_size in enumerate(output_sizes):
            if index == 0:
                if self.convex_input:
                    layers.append(
                        ConvexLinear(
                            output_size=output_size,
                            with_bias=with_bias,
                            transform=transform,
                            w_init=w_init,
                            b_init=b_init,
                            name=f"linear_{index}",
                        )
                    )
                else:
                    layers.append(
                        hk.Linear(
                            output_size=output_size,
                            with_bias=with_bias,
                            w_init=w_init,
                            b_init=b_init,
                            name=f"linear_{index}",
                        )
                    )
            elif index == (len(output_sizes) - 1):
                layers.append(
                    ConvexLinear(
                        output_size=output_size,
                        with_linear=linear_final,
                        with_bias=bias_final and bias_final,
                        transform=transform,
                        w_init=w_init,
                        b_init=b_init,
                        name=f"linear_{index}",
                    )
                )
            else:
                layers.append(
                    ConvexLinear(
                        output_size=output_size,
                        w_init=w_init,
                        b_init=b_init,
                        with_bias=with_bias,
                        transform=transform,
                        name=f"linear_{index}",
                    )
                )
        self.layers = tuple(layers)
        self.output_size = output_sizes[-1] if output_sizes else None

    def __call__(
        self,
        y: jax.Array,
        z: jax.Array | None = None,
    ) -> jax.Array:
        """
        Computes a fully input convex neural network

        Args:
        y: 入力テンソル。(*batch_size, input_size)の形状。
        z: 凸関数の出力テンソル。(*batch_size, convex_inpute_size)の形状。convex_inputがTrueのときに必要。

        Returns:
        出力テンソル。(*batch_size, output_size)の形状。yについて凸。
        """
        num_layers = len(self.layers)

        for i, layer in enumerate(self.layers):
            if i > 0 or self.convex_input:
                z = layer(z, y)
            else:
                z = layer(y)

            if i < (num_layers - 1) or self.activate_final:
                z = self.activation(z)

        return z
