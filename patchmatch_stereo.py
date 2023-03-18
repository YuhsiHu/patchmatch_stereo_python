import random
import cv2
import numpy as np
from timer import timer
from config import PMSConfig
from disparity_plane import DisparityPlane
from PVector3f import PVector3f
from propagation import PropagationPMS

class PatchMatchStereo:
    def __init__(self, width, height, config):
        self.image_left = None
        self.image_right = None
        self.width = width
        self.height = height
        self.config = config
        self.disparity_range = config.max_disparity - config.min_disparity
        self.gray_left = np.zeros([self.height, self.width], dtype=int)
        self.gray_right = np.zeros([self.height, self.width], dtype=int)
        self.grad_left = np.zeros([self.height, self.width, 2], dtype=float)
        self.grad_right = np.zeros([self.height, self.width, 2], dtype=float)
        self.cost_left = np.zeros([self.height, self.width], dtype=float)
        self.cost_right = np.zeros([self.height, self.width], dtype=float)
        self.disparity_left = np.zeros([self.height, self.width], dtype=float)
        self.disparity_right = np.zeros([self.height, self.width], dtype=float)
        self.plane_left = np.zeros([self.height, self.width], dtype=object)
        self.plane_right = np.zeros([self.height, self.width], dtype=object)
        self.mistakes_left = list()
        self.mistakes_right = list()
        self.invalid_disparity = 1024.0

    @timer("Initialize parameters")
    def random_init(self):
        for y in range(self.height):
            for x in range(self.width):
                # random disparity [min, max)
                disp_l = np.random.uniform(float(self.config.min_disparity), float(self.config.max_disparity))
                disp_r = np.random.uniform(float(self.config.min_disparity), float(self.config.max_disparity))
                if self.config.is_integer_disparity:
                    disp_l, disp_r = int(disp_l), int(disp_r)
                self.disparity_left[y, x], self.disparity_right[y, x] = disp_l, disp_r
                # random normal vector, z cannot be zero
                norm_l, norm_r = PVector3f(x=0.0, y=0.0, z=1.0), PVector3f(x=0.0, y=0.0, z=1.0)
                # is fronto-parallel windows: true or false
                if not self.config.is_force_fpw:
                    norm_l = PVector3f(
                        x=np.random.uniform(-1.0, 1.0),
                        y=np.random.uniform(-1.0, 1.0),
                        z=np.random.uniform(-1.0, 1.0),
                    )
                    norm_r = PVector3f(
                        x=np.random.uniform(-1.0, 1.0),
                        y=np.random.uniform(-1.0, 1.0),
                        z=np.random.uniform(-1.0, 1.0),
                    )
                    while norm_l.z == 0.0:
                        norm_l.z = np.random.uniform(-1.0, 1.0)
                    while norm_r.z == 0.0:
                        norm_r.z = np.random.uniform(-1.0, 1.0)
                    norm_l, norm_r = norm_l.normalize(), norm_r.normalize()
                # random disparity plane
                self.plane_left[y, x] = DisparityPlane(x=x, y=y, d=disp_l, n=norm_l)
                self.plane_right[y, x] = DisparityPlane(x=x, y=y, d=disp_r, n=norm_r)

    @timer("Total match")
    def match(self, image_left, image_right):
        self.image_left, self.image_right = image_left, image_right
        self.random_init()
        self.compute_gray()
        self.compute_gradient()
        self.propagation()
        self.plane_to_disparity()

        if self.config.is_check_lr:
            self.lr_check()
        if self.config.is_fill_holes:
            self.fill_holes_in_disparity_map()

    @timer("Propagation")
    def propagation(self):
        config_left = self.config.clone()
        config_right = self.config.clone()
        config_right.min_disparity = -config_left.max_disparity
        config_right.max_disparity = -config_left.min_disparity
        propagation_left = PropagationPMS(self.image_left, self.image_right, self.width, self.height,
                                    self.grad_left, self.grad_right, self.plane_left, self.plane_right,
                                    config_left, self.cost_left, self.cost_right, self.disparity_left)
        propagation_right = PropagationPMS(self.image_right, self.image_left, self.width, self.height,
                                    self.grad_right, self.grad_left, self.plane_right, self.plane_left,
                                    config_right, self.cost_right, self.cost_left, self.disparity_right)
        for iter in range(self.config.n_iter):
            propagation_left.do_propagation(curr_iter=iter)
            propagation_right.do_propagation(curr_iter=iter)

    @timer("Plane to disparity")
    def plane_to_disparity(self):
        for y in range(self.height):
            for x in range(self.width):
                self.disparity_left[y, x] = self.plane_left[y, x].to_disparity(x=x, y=y)
                self.disparity_right[y, x] = self.plane_right[y, x].to_disparity(x=x, y=y)

    @timer("Initialize gray")
    def compute_gray(self):
        """
        computer gray from rgb picture
        """
        for h in range(self.height):
            for w in range(self.width):
                # cv2: BGR 
                blue, green, red = self.image_left[h, w]
                self.gray_left[h, w] = int(red * 0.299 + green * 0.587 + blue * 0.114)
                blue, green, red = self.image_right[h, w]
                self.gray_right[h, w] = int(red * 0.299 + green * 0.587 + blue * 0.114)

    @timer("Initialize gradient")
    def compute_gradient(self):
        for y in range(1, self.height - 1, 1):
            for x in range(1, self.width - 1, 1):
                
                grad_x = self.gray_left[y - 1, x + 1] - self.gray_left[y - 1, x - 1] \
                         + 2 * self.gray_left[y, x + 1] - 2 * self.gray_left[y, x - 1] \
                         + self.gray_left[y + 1, x + 1] - self.gray_left[y + 1, x - 1]
                
                grad_y = self.gray_left[y + 1, x - 1] - self.gray_left[y - 1, x - 1] \
                         + 2 * self.gray_left[y + 1, x] - 2 * self.gray_left[y - 1, x] \
                         + self.gray_left[y + 1, x + 1] - self.gray_left[y - 1, x + 1]
                
                grad_y, grad_x = grad_y / 8, grad_x / 8
                
                self.grad_left[y, x, 0] = grad_x
                self.grad_left[y, x, 1] = grad_y

                grad_x = self.gray_right[y - 1, x + 1] - self.gray_right[y - 1, x - 1] \
                         + 2 * self.gray_right[y, x + 1] - 2 * self.gray_right[y, x - 1] \
                         + self.gray_right[y + 1, x + 1] - self.gray_right[y + 1, x - 1]
                grad_y = self.gray_right[y + 1, x - 1] - self.gray_right[y - 1, x - 1] \
                         + 2 * self.gray_right[y + 1, x] - 2 * self.gray_right[y - 1, x] \
                         + self.gray_right[y + 1, x + 1] - self.gray_right[y - 1, x + 1]
                grad_y, grad_x = grad_y / 8, grad_x / 8
                self.grad_right[y, x, 0] = grad_x
                self.grad_right[y, x, 1] = grad_y

    @timer("LR check")
    def lr_check(self):
        """
        left/right consistency check
        """
        for y in range(self.height):
            for x in range(self.width):
                disp = self.disparity_left[y, x]
                if disp == self.invalid_disparity:
                    self.mistakes_left.append([x, y])
                    continue
                col_right = round(x - disp)
                if 0 <= col_right < self.width:
                    disp_r = self.disparity_right[y, col_right]
                    if abs(disp + disp_r) > self.config.lr_check_threshold:
                        self.disparity_left[y, x] = self.invalid_disparity
                        self.mistakes_left.append([x, y])
                else:
                    self.disparity_left[y, x] = self.invalid_disparity
                    self.mistakes_left.append([x, y])

        for y in range(self.height):
            for x in range(self.width):
                disp = self.disparity_right[y, x]
                if disp == self.invalid_disparity:
                    self.mistakes_right.append([x, y])
                    continue
                col_right = round(x - disp)
                if 0 <= col_right < self.width:
                    disp_r = self.disparity_left[y, col_right]
                    if abs(disp + disp_r) > self.config.lr_check_threshold:
                        self.disparity_right[y, x] = self.invalid_disparity
                        self.mistakes_right.append([x, y])
                else:
                    self.disparity_right[y, x] = self.invalid_disparity
                    self.mistakes_right.append([x, y])

    @timer("Fill holes")
    def fill_holes_in_disparity_map(self):
        for i in range(len(self.mistakes_left)):
            left_planes = list()
            x, y = self.mistakes_left[i]
            xs = x + 1
            while xs < self.width:
                if self.disparity_left[y, xs] != self.invalid_disparity:
                    left_planes.append(self.plane_left[y, xs])
                    break
                xs += 1
            xs = x - 1
            while xs >= 0:
                if self.disparity_left[y, xs] != self.invalid_disparity:
                    left_planes.append(self.plane_left[y, xs])
                    break
                xs -= 1
            if len(left_planes) == 1:
                self.disparity_left[y, x] = left_planes[0].to_disparity(x=x, y=y)
            elif len(left_planes) > 1:
                d0 = left_planes[0].to_disparity(x=x, y=y)
                d1 = left_planes[1].to_disparity(x=x, y=y)
                self.disparity_left[y, x] = min(abs(d0), abs(d1))

        for i in range(len(self.mistakes_right)):
            right_planes = list()
            x, y = self.mistakes_right[i]
            xs = x + 1
            while xs < self.width:
                if self.disparity_right[y, xs] != self.invalid_disparity:
                    right_planes.append(self.plane_right[y, xs])
                    break
                xs += 1
            xs = x - 1
            while xs >= 0:
                if self.disparity_right[y, xs] != self.invalid_disparity:
                    right_planes.append(self.plane_right[y, xs])
                    break
                xs -= 1
            if len(right_planes) == 1:
                self.disparity_right[y, x] = right_planes[0].to_disparity(x=x, y=y)
            elif len(right_planes) > 1:
                d0 = right_planes[0].to_disparity(x=x, y=y)
                d1 = right_planes[1].to_disparity(x=x, y=y)
                self.disparity_right[y, x] = min(abs(d0), abs(d1))

    @timer("Get disparity map")
    def get_disparity_map(self, view=0, norm=False):
        return self._get_disparity_map(view=view, norm=norm)

    def _get_disparity_map(self, view=0, norm=False):
        if view == 0:
            disparity = self.disparity_left.copy()
        else:
            disparity = self.disparity_right.copy()
        if norm:
            disparity = np.clip(disparity, self.config.min_disparity, self.config.max_disparity)
            disparity = disparity / (self.config.max_disparity - self.config.min_disparity) * 255
        return disparity

    @timer("Get disparity cloud")
    def get_disparity_cloud(self, baseline, focal_length, principal_point_left, principal_point_right):
        b = baseline
        f = focal_length
        l_x, l_y = principal_point_left
        r_x, r_y = principal_point_right
        cloud = list()
        for y in range(self.height):
            for x in range(self.width):
                disp = np.abs(self._get_disparity_map(view=0)[y, x])
                z_ = b * f / (disp + (r_x - l_x))
                x_ = z_ * (x - l_x) / f
                y_ = z_ * (y - l_y) / f
                cloud.append([x_, y_, z_])
        return cloud


if __name__ == "__main__":
    # read left and right images
    image_left = cv2.imread("data/pms_0_left.png")
    image_right = cv2.imread("data/pms_0_right.png")
    # read config
    config_ = PMSConfig("config.json")
    # height and width
    height_, width_ = image_left.shape[0], image_left.shape[1]
    p = PatchMatchStereo(height=height_, width=width_, config=config_)
    p.match(image_left=image_left, image_right=image_right)
    disparity_ = p.get_disparity_map(view=0, norm=True)
    # write disparity to png
    cv2.imwrite("./data/pms_0_disparity.png", disparity_)
    # cloud
    cloud = p.get_disparity_cloud(
        baseline=193.001,
        focal_length=999.421,
        principal_point_left=(294.182, 252.932),
        principal_point_right=(326.95975, 252.932)
    )
    with open("./data/pms_0_clouds.txt", "w") as f:
        for c in cloud:
            f.write(" ".join([str(i) for i in c]) + "\n")