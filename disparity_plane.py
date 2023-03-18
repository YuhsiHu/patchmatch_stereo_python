from PVector3f import PVector3f

class DisparityPlane:
    """
    disparity plane
    """
    def __init__(self, x: int = 0, y: int = 0, d: int = 0, n: PVector3f = None, p: PVector3f = None):
        """
        init
        :param x: x coordinate
        :param y: y coordinate
        :param d: disparity
        :param n: normal vector
        :param p: plane vector
        """
        if p is None:
            x, y, z = -n.x / n.z, -n.y / n.z, (n.x * x + n.y * y + n.z * d) / n.z
            self.p = PVector3f(x, y, z)
        else:
            self.p = PVector3f(p.x, p.y, p.z)

    def to_disparity(self, x: int, y: int):
        """
        get disparity of (x,y) of the plane
        :param x: x coordinate
        :param y: y coordinate
        :return: disparity
        """
        return self.p * PVector3f(x, y, 1)

    def to_norm(self):
        """
        get normal vector of the plane
        :return: normal vector
        """
        return PVector3f(self.p.x, self.p.y, -1).normalize()

    def to_another_view(self, x: int, y: int):
        """
        convert to another view
        :param x: x coordinate
        :param y: y coordinate
        :return: disparity plane

        if the disparity plan of the left view is: d = a_{p}*x_{left} + b_{p}*y_{left} + c_{p}
        left and right: 
            1. x_{right} = x_{left} - d_{p}
            2. y_{right} = y_{left}
            3. disparity_{right} = -disparity(left)
        
        therefore, the disparity plane of the right view is: d = -a_{p}*x_{right} - b_{p}*y_{right} - (c_{p}+a_{p}*d_{p})
        """
        d = self.to_disparity(x, y)
        return DisparityPlane(p=PVector3f(-self.p.x, -self.p.y, -self.p.z - self.p.x * d))

    def __eq__(self, other):
        """
        equal
        """
        if not isinstance(other, DisparityPlane):
            raise TypeError(f"{type(self)} and {type(other)} could not compare.")
        return self.p == other.p
