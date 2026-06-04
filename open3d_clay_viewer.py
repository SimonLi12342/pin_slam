import argparse
import time
import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering


def make_flat_shaded_mesh(mesh: o3d.geometry.TriangleMesh):
    """
    Force flat / faceted shading by giving each triangle its own vertices.
    This makes the mesh look more like the low-poly style in your screenshot.

    Note:
    This improves the faceted visual style, but it can reduce FPS for very large meshes,
    because it duplicates vertices for every triangle.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    new_vertices = []
    new_triangles = []

    for tri in triangles:
        idx0 = len(new_vertices)
        new_vertices.append(vertices[tri[0]])
        new_vertices.append(vertices[tri[1]])
        new_vertices.append(vertices[tri[2]])
        new_triangles.append([idx0, idx0 + 1, idx0 + 2])

    flat_mesh = o3d.geometry.TriangleMesh()
    flat_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(new_vertices))
    flat_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(new_triangles))

    flat_mesh.compute_vertex_normals()
    flat_mesh.compute_triangle_normals()

    return flat_mesh


class ClayMeshViewer:
    def __init__(self, mesh_path, width=1600, height=900):
        self.mesh_path = mesh_path
        self.width = width
        self.height = height
        self.screenshot_id = 0

        self.app = gui.Application.instance
        self.app.initialize()

        self.window = self.app.create_window(
            "Open3D Clay Mesh Viewer",
            width,
            height,
        )

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene_widget)

        self.window.set_on_layout(self.on_layout)
        self.scene_widget.set_on_key(self.on_key)

        self.load_scene()

    def on_layout(self, layout_context):
        rect = self.window.content_rect
        self.scene_widget.frame = rect

    def load_scene(self):
        mesh = o3d.io.read_triangle_mesh(self.mesh_path)

        if mesh.is_empty():
            raise RuntimeError(f"Failed to read mesh: {self.mesh_path}")

        # Center the mesh around origin
        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        mesh.translate(-center)

        # Important: make faceted / low-poly look
        mesh = make_flat_shaded_mesh(mesh)

        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()
        radius = np.linalg.norm(extent)

        # Clay-like material
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"

        # Lighter mesh color.
        # Original was [0.72, 0.72, 0.72, 1.0].
        # This lighter value gives a white/gray clay style closer to your reference image.
        mat.base_color = [0.86, 0.86, 0.86, 1.0]

        mat.base_roughness = 0.9
        mat.base_metallic = 0.0
        mat.base_reflectance = 0.1

        # White background
        self.scene_widget.scene.set_background([1.0, 1.0, 1.0, 1.0])
        self.scene_widget.scene.add_geometry("mesh", mesh, mat)

        # Lighting setup
        scene = self.scene_widget.scene.scene

        # Soft global illumination-like brightness
        try:
            scene.set_indirect_light_intensity(35000)
        except Exception:
            pass

        # Sun/directional light
        sun_direction = [-0.5, -0.6, -1.0]
        sun_color = [1.0, 1.0, 1.0]
        sun_intensity = 65000

        scene.set_sun_light(sun_direction, sun_color, sun_intensity)
        scene.enable_sun_light(True)

        # Camera setup
        eye = center + np.array([0.9 * radius, -1.2 * radius, 0.45 * radius])
        target = center
        up = [0, 0, 1]

        self.scene_widget.setup_camera(60.0, bbox, center)
        self.scene_widget.look_at(target, eye, up)

        # Hide axes
        self.scene_widget.scene.show_axes(False)

        print("Open3D clay-style viewer ready.")
        print("Mouse left: rotate")
        print("Mouse wheel: zoom")
        print("Shift/Ctrl + mouse: pan, depending on your Open3D version")
        print("Press S to save screenshot")
        print("Press Q or close window to exit")

    def on_key(self, event):
        if event.type == gui.KeyEvent.DOWN:
            if event.key == gui.KeyName.S:
                self.save_screenshot()
                return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    def save_screenshot(self):
        filename = f"open3d_clay_screenshot_{self.screenshot_id:03d}.png"
        self.screenshot_id += 1

        def callback(image):
            o3d.io.write_image(filename, image)
            print(f"Saved screenshot: {filename}")

        self.window.renderer.render_to_image(callback)

    def run(self):
        self.app.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Path to .ply/.obj/.stl mesh")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    viewer = ClayMeshViewer(
        mesh_path=args.mesh,
        width=args.width,
        height=args.height,
    )
    viewer.run()


if __name__ == "__main__":
    main()