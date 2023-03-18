import numpy as np
from PVector3f import PVector3f
from disparity_plane import DisparityPlane
from compute_cost import CostComputerPMS
import time

class PropagationPMS:
    def __init__(self, image_left, image_right, width, height, grad_left, grad_right,
                 plane_left, plane_right, config, cost_left, cost_right, disparity_map):
        self.image_left = image_left
        self.image_right = image_right
        self.width = width
        self.height = height
        self.grad_left = grad_left
        self.grad_right = grad_right
        self.plane_left = plane_left
        self.plane_right = plane_right
        self.config = config
        self.cost_left = cost_left
        self.cost_right = cost_right
        self.disparity_map = disparity_map
        self.cost_cpt_left = CostComputerPMS(image_left, image_right, grad_left, grad_right, width, height,
                                             config.patch_size, config.min_disparity, config.max_disparity,
                                             config.gamma, config.alpha, config.tau_col, config.tau_grad)
        self.cost_cpt_right = CostComputerPMS(image_right, image_left, grad_right, grad_left, width, height,
                                             config.patch_size, -config.max_disparity, -config.min_disparity,
                                             config.gamma, config.alpha, config.tau_col, config.tau_grad)
        self.compute_cost_data()

    def do_propagation(self, curr_iter):
        """
        performs one iteration by propagating the disparity estimates from neighboring pixels in both spatial and view directions.
        :param curr_iter: current iteration
        :return: None
        """
        # 根据迭代次数确定空间传播的方向，0 代表从上到下，1 代表从下到上
        direction = 1 if curr_iter % 2 == 0 else -1
        # 根据迭代次数确定起始点的纵坐标
        y = 0 if curr_iter % 2 == 0 else self.height - 1
        count_ = 0
        times_ = 0
        print(f"\r| Propagation iter {curr_iter}: 0", end="")
        # 遍历所有像素进行空间和视角传播
        for i in range(self.height):
            # 根据当前迭代次数确定起始点的横坐标
            x = 0 if curr_iter % 2 == 0 else self.width - 1
            for j in range(self.width):
                start_ = time.time()
                # 空间传播
                self.spatial_propagation(x=x, y=y, direction=direction)
                # 平面优化
                if not self.config.is_force_fpw:
                    self.plane_refine(x=x, y=y)
                # 视角传播
                self.view_propagation(x=x, y=y)
                x += direction
                times_ += time.time() - start_
                count_ += 1
                print(f"\r| Propagation iter {curr_iter}: [{y * self.width + x + 1} / {self.height * self.width}] {times_:.0f}s/{times_ / count_ * (self.width * self.height - count_):.0f}s, {count_/times_:.3f} it/s", end="")
            y += direction
        print(f"\r| Propagation iter {curr_iter} cost {times_:.3f} seconds.")

    def compute_cost_data(self):
        """
        compute the cost data for the left and right images
        :return: None"""
        print(f"\r| Init cost {0} / {self.height * self.width}", end="")
        count_ = 0
        times_ = 0
        for y in range(self.height):
            for x in range(self.width):
                start_ = time.time()
                p = self.plane_left[y, x]
                self.cost_left[y, x] = self.cost_cpt_left.compute_agg(x=x, y=y, p=p)
                times_ += time.time() - start_
                count_ += 1
                print(f"\r| Initialize cost [{y * self.width + x + 1} / {self.height * self.width}] {times_:.0f}s/{times_ / count_ * (self.width * self.height - count_):.0f}s, {count_/times_:.3f} it/s", end="")
        print(f"\r| Initialize cost {times_:.3f} seconds.")

    def spatial_propagation(self, x, y, direction):
        """
        propagates the disparity estimates from neighboring pixels in the spatial direction.
        :param x: x coordinate of the pixel
        :param y: y coordinate of the pixel
        :param direction: propagation direction
        :return: None
        """
        plane_p = self.plane_left[y, x]
        cost_p = self.cost_left[y, x]
        # 获取p左(右)侧像素的视差平面，计算将平面分配给p时的代价，取较小值
        xd = x - direction
        if 0 <= xd < self.width:
            plane = self.plane_left[y, xd]
            if plane != plane_p:
                cost = self.cost_cpt_left.compute_agg(x=x, y=y, p=plane)
                if cost < cost_p:
                    plane_p = plane
                    cost_p = cost

        # 获取p上(下)侧像素的视差平面，计算将平面分配给p时的代价，取较小值
        yd = y - direction
        if 0 <= yd < self.height:
            plane = self.plane_left[yd, x]
            if plane != plane_p:
                cost = self.cost_cpt_left.compute_agg(x=x, y=y, p=plane)
                if cost < cost_p:
                    plane_p = plane
                    cost_p = cost

        self.plane_left[y, x] = plane_p
        self.cost_left[y, x] = cost_p

    def view_propagation(self, x, y):
        """
        propagates in the view direction
        搜索p在右视图的同名点q，更新q的平面
        """
        # 左视图匹配点p的位置及其视差平面 
        plane_p = self.plane_left[y, x]
        d_p = plane_p.to_disparity(x=x, y=y)
        # 计算右视图列号
        xr = int(x - d_p)
        if xr < 0 or xr > self.width - 1:
            return
        
        plane_q = self.plane_right[y, xr]
        cost_q = self.cost_right[y, xr]

        # 将左视图的视差平面转换到右视图
        plane_p2q = plane_p.to_another_view(x=x, y=y)
        d_q = plane_p2q.to_disparity(x=xr, y=y)
        cost = self.cost_cpt_right.compute_agg(x=xr, y=y, p=plane_p2q)
        if cost < cost_q:
            plane_q = plane_p2q
            cost_q = cost
        self.plane_right[y, xr] = plane_q
        self.cost_right[y, xr] = cost_q

    def plane_refine(self, x, y):
        """
        refine the disparity
        """
        min_disp = self.config.min_disparity
        max_disp = self.config.max_disparity
        # 像素p的平面、代价、视差、法线
        plane_p = self.plane_left[y, x]
        cost_p = self.cost_left[y, x]
        d_p = plane_p.to_disparity(x=x, y=y)
        norm_p = plane_p.to_norm()

        disp_update = (max_disp - min_disp) / 2.0
        norm_update = 1.0
        stop_thres = 0.1
        
        # iterate to refine
        while disp_update > stop_thres:
            # 在 -disp_update ~ disp_update 范围内随机一个视差增量
            disp_rd = np.random.uniform(-1.0, 1.0) * disp_update
            if self.config.is_integer_disparity:
                disp_rd = int(disp_rd)
            d_p_new = d_p + disp_rd
            # compute the new disparity
            if d_p_new < min_disp or d_p_new > max_disp:
                disp_update /= 2
                norm_update /= 2
                continue
            
            # 在 -norm_update ~ norm_update 范围内随机三个值作为法线增量的三个分量
            if not self.config.is_force_fpw:
                norm_rd = PVector3f(
                    x=np.random.uniform(-1.0, 1.0) * norm_update,
                    y=np.random.uniform(-1.0, 1.0) * norm_update,
                    z=np.random.uniform(-1.0, 1.0) * norm_update,
                )
                while norm_rd.z == 0.0:
                    norm_rd.z = np.random.uniform(-1.0, 1.0)
            else:
                norm_rd = PVector3f(x=0.0, y=0.0, z=0.0)

            # 计算像素p新的法线
            norm_p_new = norm_p + norm_rd
            norm_p_new.normalize()
            # 计算新的视差平面
            plane_new = DisparityPlane(x=x, y=y, d=d_p_new, n=norm_p_new)
            # compare the cost
            if plane_new != plane_p:
                cost = self.cost_cpt_left.compute_agg(x=x, y=y, p=plane_new)
                if cost < cost_p:
                    plane_p = plane_new
                    cost_p = cost
                    d_p = d_p_new
                    norm_p = norm_p_new
                    self.plane_left[y, x] = plane_p
                    self.cost_left[y, x] = cost_p
            disp_update /= 2.0
            norm_update /= 2.0