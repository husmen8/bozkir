"""Replay the viewer's cell draw-order logic offline.

Mirrors web/viewer.js frame(): camera basis, per-cell `near` key taken as the
minimum projected depth over the four footprint corners, tiebreak on `side`,
sorted far-to-near. Goal: find which cell pairs swap draw order between two
nearby azimuths, and where their shared boundary lands on screen.
"""
import math

TILE = 1.0
GRID = 9
FOV = 60.0
W, H = 1280, 1280


def basis(az_deg, el_deg, dist, target=(0, 0, 0)):
    az, el = math.radians(az_deg), math.radians(el_deg)
    eye = (target[0] + dist * math.cos(el) * math.cos(az),
           target[1] + dist * math.cos(el) * math.sin(az),
           target[2] + dist * math.sin(el))
    f = [target[i] - eye[i] for i in range(3)]
    n = math.hypot(*f)
    f = [v / n for v in f]
    up = (1, 0, 0) if abs(f[2]) > 0.999 else (0, 0, 1)
    r = [f[1] * up[2] - f[2] * up[1],
         f[2] * up[0] - f[0] * up[2],
         f[0] * up[1] - f[1] * up[0]]
    rl = math.hypot(*r) or 1.0
    r = [v / rl for v in r]
    d = [f[1] * r[2] - f[2] * r[1],
         f[2] * r[0] - f[0] * r[2],
         f[0] * r[1] - f[1] * r[0]]
    return dict(right=r, down=d, forward=f, eye=eye)


def cells():
    half = (GRID - 1) / 2
    out = []
    for j in range(GRID):
        for i in range(GRID):
            out.append(dict(i=i, j=j, x=(i - half) * TILE, y=(j - half) * TILE, z=0.0))
    return out


def draw_order(b, cs):
    half = TILE / 2
    vis = []
    for c in cs:
        dx, dy, dz = c['x'] - b['eye'][0], c['y'] - b['eye'][1], c['z'] - b['eye'][2]
        near = math.inf
        for ox, oy in ((-half, -half), (half, -half), (-half, half), (half, half)):
            kx, ky = dx + ox, dy + oy
            kz = kx * b['forward'][0] + ky * b['forward'][1] + dz * b['forward'][2]
            near = min(near, kz)
        sx = dx * b['right'][0] + dy * b['right'][1] + dz * b['right'][2]
        sy = dx * b['down'][0] + dy * b['down'][1] + dz * b['down'][2]
        vis.append(dict(c=c, near=near, side=abs(sx) + abs(sy)))
    vis.sort(key=lambda v: (-v['near'], -v['side']))
    return {(v['c']['i'], v['c']['j']): k for k, v in enumerate(vis)}


def topo_order(b, cs):
    """Draw order from pairwise boundary constraints. Mirrors web/order.js.

    Each adjacent pair is decided by the sign of the eye's distance to their
    shared boundary plane, which is well defined even when the two cells have
    identical depth - the case the `near` key cannot handle. The depth key is
    demoted to a tiebreak among cells that share no boundary.
    """
    idx = {(c['i'], c['j']): k for k, c in enumerate(cs)}
    n = len(cs)
    after = [[] for _ in range(n)]
    indeg = [0] * n
    for c, nb, mid, nrm in neighbours(cs):
        a, bb = idx[(c['i'], c['j'])], idx[(nb['i'], nb['j'])]
        s = sum(nrm[i] * (b['eye'][i] - mid[i]) for i in range(3))
        if s == 0.0:
            continue                       # undecidable; leave it to merging
        u, v = (a, bb) if s > 0 else (bb, a)
        after[u].append(v)
        indeg[v] += 1

    half = TILE / 2
    key = []
    for c in cs:
        dx, dy, dz = c['x'] - b['eye'][0], c['y'] - b['eye'][1], c['z'] - b['eye'][2]
        key.append(min((dx + ox) * b['forward'][0] + (dy + oy) * b['forward'][1]
                       + dz * b['forward'][2]
                       for ox, oy in ((-half, -half), (half, -half),
                                      (-half, half), (half, half))))

    done, out = [False] * n, []
    while len(out) < n:
        best = -1
        for k in range(n):
            if done[k] or indeg[k] > 0:
                continue
            if best < 0 or key[k] > key[best]:
                best = k
        if best < 0:                        # cycle: release the furthest
            best = max((k for k in range(n) if not done[k]), key=lambda k: key[k])
            indeg[best] = 0
        done[best] = True
        out.append(best)
        for v in after[best]:
            indeg[v] = max(0, indeg[v] - 1)
    return {(cs[k]['i'], cs[k]['j']): r for r, k in enumerate(out)}


def project(b, p):
    fy = (H / 2) / math.tan(math.radians(FOV) / 2)
    d = [p[i] - b['eye'][i] for i in range(3)]
    z = sum(d[i] * b['forward'][i] for i in range(3))
    if z <= 0.01:
        return None
    sx = sum(d[i] * b['right'][i] for i in range(3))
    sy = sum(d[i] * b['down'][i] for i in range(3))
    return (W / 2 + fy * sx / z, H / 2 + fy * sy / z)


def neighbours(cs):
    """Adjacent pairs plus their shared boundary midpoint and plane normal."""
    byij = {(c['i'], c['j']): c for c in cs}
    out = []
    for c in cs:
        for di, dj, nx, ny in ((1, 0, 1.0, 0.0), (0, 1, 0.0, 1.0)):
            nb = byij.get((c['i'] + di, c['j'] + dj))
            if not nb:
                continue
            mid = ((c['x'] + nb['x']) / 2, (c['y'] + nb['y']) / 2, 0.0)
            out.append((c, nb, mid, (nx, ny, 0.0)))
    return out


def run(az0, daz, el, dist, order=draw_order):
    cs = cells()
    b0, b1 = basis(az0, el, dist), basis(az0 + daz, el, dist)
    o0, o1 = order(b0, cs), order(b1, cs)
    pairs = neighbours(cs)
    flips = []
    for c, nb, mid, n in pairs:
        a, bb = (c['i'], c['j']), (nb['i'], nb['j'])
        s0 = o0[a] < o0[bb]
        s1 = o1[a] < o1[bb]
        if s0 != s1:
            scr = project(b0, mid)
            if scr and 0 <= scr[0] < W and 0 <= scr[1] < H:
                # edge-on-ness: |n . (eye - edge)|, GSWT's merge criterion
                v = [b0['eye'][i] - mid[i] for i in range(3)]
                dot = abs(sum(n[i] * v[i] for i in range(3)))
                flips.append((scr[1], scr[0], a, bb, dot, n))
    return flips, b0


if __name__ == '__main__':
    print('pairs whose draw order reverses over a 3 degree turn\n')
    print(f"{'az':>6}  {'near+side':>9}  {'topological':>11}   flipped boundaries")
    for az in (0, 5, 10, 22.5, 35, 45, 89, 90):
        a, b0 = run(az, 3.0, el=12.0, dist=6.0)
        t, _ = run(az, 3.0, el=12.0, dist=6.0, order=topo_order)
        def span(fs):
            if not fs:
                return ''
            d = sorted(f[4] for f in fs)
            return f'{d[0]:.2f}..{d[-1]:.2f}'
        print(f'{az:6.1f}  {len(a):9d}  {len(t):11d}   '
              f'|n.(eye-edge)|  key {span(a) or "-":>12}  '
              f'topo {span(t) or "-":>12}')
    print('\nThe surviving flips are the boundaries the camera stands in the')
    print('plane of, where no order is right and GSWT merges instead.')