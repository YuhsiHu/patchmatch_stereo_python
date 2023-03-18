class PVector3f:
    def __init__(self, x: float, y: float, z: float):
        """
        init
        :param x: x coordinate
        :param y: y coordinate
        :param z: z coordinate
        """
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def normalize(self):
        """
        normalize to unit vector
        :return: self
        """
        denominator = (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5
        self.x /= denominator
        self.y /= denominator
        self.z /= denominator
        return self

    def __mul__(self, other):
        """
        dot multiply
        :param other: PVector3f
        :return: PVector3f
        """
        if not isinstance(other, PVector3f):
            raise TypeError(f"{type(self)} and {type(other)} could not multiply.")
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __add__(self, other):
        """
        add
        """
        if not isinstance(other, PVector3f):
            raise TypeError(f"{type(self)} and {type(other)} could not add.")
        return PVector3f(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        """
        subtract
        """
        if not isinstance(other, PVector3f):
            raise TypeError(f"{type(self)} and {type(other)} could not sub.")
        return PVector3f(self.x - other.x, self.y - other.y, self.z - other.z)

    def __invert__(self):
        """
        invert
        """
        return PVector3f(-self.x, -self.y, -self.z)

    def __eq__(self, other):
        """
        equal
        """
        if not isinstance(other, PVector3f):
            raise TypeError(f"{type(self)} and {type(other)} could not compare.")
        return self.x == other.x and self.y == other.y and self.z == other.z