# Based on:
# - https://www.redblobgames.com/grids/hexagons/#conversions, except that (u,v)=(q,-r), and
# - https://www.redblobgames.com/grids/hexagons/#rotation

class Axial:
    __slots__ = ['u', 'v']

    def __init__(s, u, v):
        s.u, s.v = u, v

    def __eq__(s, o):
        return s.u == o.u and s.v == o.v

    def __iter__(s):
        for v in (s.u, s.v):
            yield v

    def __repr__(s):
        return f'Axial({s.u!r}, {s.v!r})'

    def toCube(s):
        x, z = s.u, -s.v
        y = -x-z
        return Cube(x, y, z) 

    def rotate_60ccw(s, n=1):
        return s.toCube().rotate_60ccw(n).toAxial()

class Cube:
    __slots__ = ['x', 'y', 'z']

    def __init__(s, x, y, z):
        s.x, s.y, s.z = x, y, z
    
    def __eq__(s, o):
        return s.x == o.x and s.y == o.y and s.z == o.z

    def __iter__(s):
        for v in (s.x, s.y, s.z):
            yield v

    def __repr__(s):
        return f'Cube({s.x!r}, {s.y!r}, {s.z!r})'

    def toAxial(s):
        return Axial(s.x, -s.z)

    def rotate_60ccw(s, n=1):
        for _ in range(n):
            s.x, s.y, s.z = -s.y, -s.z, -s.x
        return s
