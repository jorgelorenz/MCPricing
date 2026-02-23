import numpy as np

class Matrix2D:
    """Fast 2D matrix with bilinear interpolation."""

    def __init__(self, x_axis, y_axis, values):
        self.x_axis = np.asarray(x_axis, dtype=float)
        self.y_axis = np.asarray(y_axis, dtype=float)
        self.values = np.asarray(values, dtype=float)

        if self.x_axis.ndim != 1 or self.y_axis.ndim != 1:
            raise ValueError("x_axis and y_axis must be 1D arrays")
        if self.values.shape != (self.x_axis.size, self.y_axis.size):
            raise ValueError(
                "values shape must be (len(x_axis), len(y_axis))"
            )
        if np.any(np.diff(self.x_axis) <= 0) or np.any(np.diff(self.y_axis) <= 0):
            raise ValueError("x_axis and y_axis must be strictly increasing")

    def interpolate(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x, y = np.broadcast_arrays(x, y)

        x_flat = x.ravel()
        y_flat = y.ravel()

        x_clipped = np.clip(x_flat, self.x_axis[0], self.x_axis[-1])
        y_clipped = np.clip(y_flat, self.y_axis[0], self.y_axis[-1])

        ix = np.searchsorted(self.x_axis, x_clipped, side="right") - 1
        iy = np.searchsorted(self.y_axis, y_clipped, side="right") - 1

        ix = np.clip(ix, 0, self.x_axis.size - 2)
        iy = np.clip(iy, 0, self.y_axis.size - 2)

        x0 = self.x_axis[ix]
        x1 = self.x_axis[ix + 1]
        y0 = self.y_axis[iy]
        y1 = self.y_axis[iy + 1]

        tx = (x_clipped - x0) / (x1 - x0)
        ty = (y_clipped - y0) / (y1 - y0)

        v00 = self.values[ix, iy]
        v10 = self.values[ix + 1, iy]
        v01 = self.values[ix, iy + 1]
        v11 = self.values[ix + 1, iy + 1]

        out = (
            (1.0 - tx) * (1.0 - ty) * v00
            + tx * (1.0 - ty) * v10
            + (1.0 - tx) * ty * v01
            + tx * ty * v11
        )

        return out.reshape(x.shape)

    def __call__(self, x, y):
        return self.interpolate(x, y)


class VolatilityMatrix(Matrix2D):
    """Volatility matrix over (tenor, strike) axes."""

    pass


class LocalVolatilityMatrix(VolatilityMatrix):
    """Local volatility matrix with floor and a vectorized interpolator."""

    def __init__(self, tenors, strikes, local_vol_matrix, vol_floor=1e-8):
        super().__init__(tenors, strikes, local_vol_matrix)
        self.vol_floor = float(vol_floor)

    def local_vol(self, t, s):
        lv = super().interpolate(t, s)
        return np.maximum(lv, self.vol_floor)

    def get_interpolator(self):
        def _interpolator(points):
            if isinstance(points, tuple) and len(points) == 2:
                return self.local_vol(points[0], points[1])

            points = np.asarray(points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError("points must be tuple(t, s) or array with shape (n, 2)")
            return self.local_vol(points[:, 0], points[:, 1])

        return _interpolator

    def __call__(self, t, s):
        return self.local_vol(t, s)
