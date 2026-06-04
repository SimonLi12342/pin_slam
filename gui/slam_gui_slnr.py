#!/usr/bin/env python3
# @file      slam_gui_slnr.py

import open3d as o3d

from gui.gui_utils_slnr import ControlPacket
from gui.slam_gui import SLAM_GUI


class SLNR_GUI(SLAM_GUI):
    def init_widget(self):
        super().init_widget()
        max_query_nn = max(1, int(getattr(self.config, "query_nn_k", 1)))
        self.mesh_min_nn_slider.set_limits(1, max_query_nn)
        self.mesh_min_nn_slider.int_value = min(
            max(1, int(self.mesh_min_nn_slider.int_value)),
            max_query_nn,
        )

    def send_data(self):
        packet = ControlPacket()
        packet.flag_pause = not self.slider_slam.is_on
        packet.flag_vis = self.slider_vis.is_on
        packet.flag_source = self.scan_regis_color_chbox.checked
        packet.flag_mesh = self.mesh_chbox.checked
        packet.flag_sdf = self.sdf_chbox.checked
        packet.flag_global = not self.local_map_chbox.checked
        packet.mc_res_m = self.mesh_mc_res_slider.int_value / 100.0
        packet.mesh_min_nn = min(
            max(1, self.mesh_min_nn_slider.int_value),
            max(1, int(getattr(self.config, "query_nn_k", 1))),
        )
        packet.mesh_freq_frame = self.mesh_freq_frame_slider.int_value
        packet.sdf_freq_frame = self.sdf_freq_frame_slider.int_value
        packet.sdf_slice_height = self.sdf_slice_height_slider.double_value
        packet.sdf_res_m = self.sdf_res_slider.int_value / 100.0
        packet.cur_frame_id = self.cur_frame_id
        self.q_vis2main.put(packet)

    def receive_data(self, q):
        super().receive_data(q)
        data_packet = self.cur_data_packet
        if data_packet is None or not data_packet.has_neural_points:
            return
        self.neural_points_info.text = "# Neural points: {} (local {}) [Map size: {:.1f} MB]".format(
            data_packet.neural_points_data["count"],
            data_packet.neural_points_data["local_count"],
            data_packet.neural_points_data["map_memory_mb"],
        )


def run(params_gui=None):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    SLNR_GUI(params_gui)
    app.run()


def main():
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    SLNR_GUI()
    app.run()


if __name__ == "__main__":
    main()
