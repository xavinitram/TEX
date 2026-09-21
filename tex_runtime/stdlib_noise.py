"""
TEX Standard Library — procedural-noise builtins (one domain leaf of `stdlib.py`).

The `TEXStdlib` class-body section(s) Noise functions moved here verbatim, onto the `_StdlibNoise`
mixin. `stdlib.py` composes the leaves' mixins into `TEXStdlib` in the class body's original
section order, which is the REG-1 registration order (`help_entries()`, the generated
reference and the help panel all read it) — so import this leaf THROUGH `stdlib`, not
directly, unless registering only this domain is what you want.
"""
from __future__ import annotations
import torch
from .stdlib_registry import stdlib
from .stdlib_core import (
    _to_float,
    _to_tensor,
)
# The noise kernels live in noise.py; the builtins below are their TEX-facing wrappers.
from .noise import (
    _perlin2d_fast, _perlin3d_fast, _simplex2d,
    _fbm2d, _fbm3d,
    _worley2d, _worley3d,
    _worley2d_id, _worley3d_id,
    _curl2d, _curl3d,
    _ridged2d, _ridged3d,
    _billow2d, _billow3d,
    _turbulence2d, _turbulence3d,
    _flow2d, _flow3d,
    _alligator2d, _alligator3d,
)

# `TEXStdlib` is the class `stdlib.py` composes from every leaf. A leaf cannot import it at
# load time (the facade imports the leaves), so the facade BINDS it into this namespace the
# moment the class exists; the `TEXStdlib.fn_*(...)` delegations below then resolve at call
# time exactly as they did inside the one-file class. The spelling is load-bearing:
# `stdlib_registry._impl_looks_fragile` reads the literal `TEXStdlib.fn_*(` from the source
# to follow one level of delegation, so it must not be rewritten to the mixin's name.
TEXStdlib = None


class _StdlibNoise:
    """procedural-noise builtins: the `fn_*` methods `stdlib.py` mixes into `TEXStdlib`."""

    # -- Noise functions ------------------------------------------------
    # All noise functions support optional z for 3D:
    #   2 args = 2D, 3 args = 3D (for base noise)
    #   3 args = 2D with octaves, 4 args = 3D with octaves (for FBM family)

    @stdlib("perlin", sig='perlin(x, y) \\u2192 float', category='Noise', doc='2D Perlin noise. Returns value in [-1, 1].', ex='float n = perlin(u * 10.0, v * 10.0);')
    @staticmethod
    def fn_perlin(x, y, z=None) -> torch.Tensor:
        """Perlin noise. 2D when z omitted, 3D when z provided. Returns float in ~[-1, 1]."""
        if z is not None:
            return _perlin3d_fast(_to_tensor(x), _to_tensor(y), _to_tensor(z))
        return _perlin2d_fast(_to_tensor(x), _to_tensor(y))

    @stdlib("simplex", sig='simplex(x, y) \\u2192 float', category='Noise', doc='2D Simplex noise. Returns value in [-1, 1].', ex='float n = simplex(u * 8.0, v * 8.0);')
    @staticmethod
    def fn_simplex(x, y, z=None) -> torch.Tensor:
        """Simplex noise. 2D when z omitted, 3D falls back to Perlin. Returns float in ~[-1, 1]."""
        if z is not None:
            return _perlin3d_fast(_to_tensor(x), _to_tensor(y), _to_tensor(z))
        return _simplex2d(_to_tensor(x), _to_tensor(y))

    @stdlib("fbm", sig='fbm(x, y, octaves) \\u2192 float', category='Noise', sync=True, doc='Fractal Brownian Motion (multi-octave Perlin).', ex='float n = fbm(u * 4.0, v * 4.0, 6);')
    @staticmethod
    def fn_fbm(x, y, z_or_oct, octaves=None) -> torch.Tensor:
        """FBM noise. fbm(x,y,octaves) for 2D, fbm(x,y,z,octaves) for 3D."""
        if octaves is not None:
            return _fbm3d(_to_tensor(x), _to_tensor(y), _to_tensor(z_or_oct),
                          int(_to_float(octaves)))
        return _fbm2d(_to_tensor(x), _to_tensor(y), int(_to_float(z_or_oct)))

    @stdlib("worley_f1", sig='worley_f1(x, y) \\u2192 float', category='Noise', doc='Worley (cellular) noise — distance to nearest cell center.', ex='float n = worley_f1(u * 5.0, v * 5.0);')
    @staticmethod
    def fn_worley_f1(x, y, z=None) -> torch.Tensor:
        """Worley F1 noise (nearest cell distance). Returns float in ~[0, 1]."""
        if z is not None:
            return _worley3d(_to_tensor(x), _to_tensor(y), _to_tensor(z), return_f2=False)
        return _worley2d(_to_tensor(x), _to_tensor(y), return_f2=False)

    @stdlib("worley_f2", sig='worley_f2(x, y) \\u2192 float', category='Noise', doc='Worley noise — distance to second-nearest cell center.', ex='float n = worley_f2(u * 5.0, v * 5.0);')
    @staticmethod
    def fn_worley_f2(x, y, z=None) -> torch.Tensor:
        """Worley F2 noise (2nd nearest cell distance). Returns float in ~[0, 1]."""
        if z is not None:
            return _worley3d(_to_tensor(x), _to_tensor(y), _to_tensor(z), return_f2=True)
        return _worley2d(_to_tensor(x), _to_tensor(y), return_f2=True)

    @stdlib("voronoi", sig='voronoi(x, y) \\u2192 float', category='Noise', doc='Alias of worley_f1 — distance to the nearest feature point. For a per-cell value use worley_id.', ex='float d = voronoi(u * 8.0, v * 8.0);')
    @staticmethod
    def fn_voronoi(x, y, z=None) -> torch.Tensor:
        """Voronoi noise (alias for worley_f1). Unchanged output — only the help
        text above was wrong (it claimed a per-cell id; this is a distance)."""
        return TEXStdlib.fn_worley_f1(x, y, z)

    # ASK-5: a per-cell id, distinct from worley_f1/f2's distances above. Footprint
    # 'point' (reads only its own coordinate args); no sync (single-eval, capturable);
    # not spatial (no stencil emitter — codegen falls through to the general dispatch,
    # which calls this exact object, so interp/codegen agree by construction). Runs
    # eager on every tier deliberately — see noise._worley2d_id's docstring for why it
    # never touches the noise cache's eager->traced->compiled promotion path.
    @stdlib("worley_id", sig='worley_id(x, y) \\u2192 float', category='Noise', doc="Worley cell id: a stable value in [0, 1] per cell of worley_f1's nearest feature point.", ex='float id = worley_id(u * 8.0, v * 8.0);')
    @staticmethod
    def fn_worley_id(x, y, z=None) -> torch.Tensor:
        """Worley per-cell id (hash of the nearest cell). Returns float in [0, 1].

        2D when z is omitted, 3D when provided — same arity convention as
        worley_f1/f2. Always eager: `_worley2d_id`/`_worley3d_id` duplicate the
        core neighbour search rather than sharing `_worley2d`/`_worley3d`'s
        `_TieredCache`-backed path, so this never gets traced or torch.compiled.
        """
        if z is not None:
            return _worley3d_id(_to_tensor(x), _to_tensor(y), _to_tensor(z))
        return _worley2d_id(_to_tensor(x), _to_tensor(y))

    @stdlib("curl", sig='curl(x, y) \\u2192 vec2', category='Noise', doc='Curl of 2D noise field. Returns a divergence-free vector.', ex='vec2 c = curl(u * 5.0, v * 5.0);')
    @staticmethod
    def fn_curl(x, y, z=None) -> torch.Tensor:
        """Curl noise. 2D → vec2 (divergence-free), 3D → vec3."""
        if z is not None:
            return _curl3d(_to_tensor(x), _to_tensor(y), _to_tensor(z))
        return _curl2d(_to_tensor(x), _to_tensor(y))

    @stdlib("ridged", sig='ridged(x, y, octaves) \\u2192 float', category='Noise', sync=True, doc='Ridged multifractal — sharp ridges, good for mountains.', ex='float n = ridged(u * 4.0, v * 4.0, 6);')
    @staticmethod
    def fn_ridged(x, y, z_or_oct, octaves=None) -> torch.Tensor:
        """Ridged FBM. ridged(x,y,octaves) for 2D, ridged(x,y,z,octaves) for 3D."""
        if octaves is not None:
            return _ridged3d(_to_tensor(x), _to_tensor(y), _to_tensor(z_or_oct),
                             int(_to_float(octaves)))
        return _ridged2d(_to_tensor(x), _to_tensor(y), int(_to_float(z_or_oct)))

    @stdlib("billow", sig='billow(x, y, octaves) \\u2192 float', category='Noise', sync=True, doc='Billowy noise — abs(fbm). Puffy cloud shapes.', ex='float n = billow(u * 4.0, v * 4.0, 6);')
    @staticmethod
    def fn_billow(x, y, z_or_oct, octaves=None) -> torch.Tensor:
        """Billow FBM. billow(x,y,octaves) for 2D, billow(x,y,z,octaves) for 3D."""
        if octaves is not None:
            return _billow3d(_to_tensor(x), _to_tensor(y), _to_tensor(z_or_oct),
                             int(_to_float(octaves)))
        return _billow2d(_to_tensor(x), _to_tensor(y), int(_to_float(z_or_oct)))

    @stdlib("turbulence", sig='turbulence(x, y, octaves) \\u2192 float', category='Noise', sync=True, doc='Turbulence — sum of abs(noise) per octave. Veiny patterns.', ex='float n = turbulence(u * 4.0, v * 4.0, 6);')
    @staticmethod
    def fn_turbulence(x, y, z_or_oct, octaves=None) -> torch.Tensor:
        """Turbulence. turbulence(x,y,octaves) for 2D, turbulence(x,y,z,octaves) for 3D."""
        if octaves is not None:
            return _turbulence3d(_to_tensor(x), _to_tensor(y), _to_tensor(z_or_oct),
                                 int(_to_float(octaves)))
        return _turbulence2d(_to_tensor(x), _to_tensor(y), int(_to_float(z_or_oct)))

    @stdlib("flow", sig='flow(x, y, angle) \\u2192 float', category='Noise', sync=True, doc='Flow noise — Perlin rotated by angle per octave. Avoids static patterns.', ex='float n = flow(u * 6.0, v * 6.0, fi * 0.1);')
    @staticmethod
    def fn_flow(x, y, z_or_time, time=None) -> torch.Tensor:
        """Flow noise. flow(x,y,time) for 2D, flow(x,y,z,time) for 3D."""
        if time is not None:
            return _flow3d(_to_tensor(x), _to_tensor(y), _to_tensor(z_or_time),
                           _to_float(time))
        return _flow2d(_to_tensor(x), _to_tensor(y), _to_float(z_or_time))

    @stdlib("alligator", sig='alligator(x, y) \\u2192 float', category='Noise', sync=True, doc='Alligator noise — cellular crack patterns.', ex='float n = alligator(u * 5.0, v * 5.0);')
    @staticmethod
    def fn_alligator(x, y, z_or_oct=None, octaves=None) -> torch.Tensor:
        """Alligator noise. 2 args: 2D default octaves; 3 args: 2D custom octaves; 4 args: 3D."""
        if octaves is not None:
            return _alligator3d(_to_tensor(x), _to_tensor(y), _to_tensor(z_or_oct),
                                int(_to_float(octaves)))
        if z_or_oct is not None:
            return _alligator2d(_to_tensor(x), _to_tensor(y), int(_to_float(z_or_oct)))
        return _alligator2d(_to_tensor(x), _to_tensor(y), 4)
