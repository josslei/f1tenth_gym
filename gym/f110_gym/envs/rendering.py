# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
Rendering engine for f1tenth gym env based on pyglet and OpenGL
Author: Hongrui Zheng
"""

# other
import numpy as np

# opengl stuff
import pyglet
import yaml

# helpers
from f110_gym.envs.collision_models import get_vertices
from PIL import Image
from pyglet import gl
from pyglet.graphics import get_default_shader
from pyglet.math import Mat4

# zooming constants
ZOOM_IN_FACTOR = 1.2
ZOOM_OUT_FACTOR = 1 / ZOOM_IN_FACTOR

# vehicle shape constants
CAR_LENGTH = 0.58
CAR_WIDTH = 0.31
CAR_TRIANGLE_INDICES = (0, 1, 2, 0, 2, 3)
RENDER_SCALE = 50.0


def _flatten_vertices(vertices: np.ndarray) -> list[float]:
    if vertices.shape[1] == 2:
        zeros = np.zeros((vertices.shape[0], 1), dtype=vertices.dtype)
        vertices = np.column_stack((vertices, zeros))
    return vertices.flatten().tolist()


class EnvRenderer(pyglet.window.Window):
    """
    A window class inherited from pyglet.window.Window, handles the camera/projection interaction, resizing window, and rendering the environment
    """

    def __init__(self, width, height, *args, **kwargs):
        """
        Class constructor

        Args:
            width (int): width of the window
            height (int): height of the window

        Returns:
            None
        """
        conf = gl.Config(sample_buffers=1, samples=4, depth_size=16, double_buffer=True)
        super().__init__(
            width, height, config=conf, resizable=True, vsync=False, *args, **kwargs
        )

        # gl init
        gl.glClearColor(9 / 255, 32 / 255, 87 / 255, 1.0)  # pyright: ignore[reportPrivateImportUsage]

        # initialize camera values
        self.left = -width / 2
        self.right = width / 2
        self.bottom = -height / 2
        self.top = height / 2
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.zoom_level = 1.2
        self.zoomed_width = width
        self.zoomed_height = height

        # current batch that keeps track of all graphics
        self.batch = pyglet.graphics.Batch()
        self.program = get_default_shader()

        # current env map
        self.map_points = None
        self.map_vertices = None

        # current env agent poses, (num_agents, 3), columns are (x, y, theta)
        self.poses = None

        # current env agent vertices, (num_agents, 4, 2), 2nd and 3rd dimensions are the 4 corners in 2D
        self.vertices = None
        self.cars = []

        # current score label
        self.score_label = pyglet.text.Label(
            "Lap Time: {laptime:.2f}, Ego Lap Count: {count:.0f}".format(
                laptime=0.0, count=0.0
            ),
            font_size=36,
            x=0,
            y=-800,
            anchor_x="center",
            anchor_y="center",
            # width=0.01,
            # height=0.01,
            color=(255, 255, 255, 255),
            batch=self.batch,
        )

        self.window_closed = False
        self.fps_display = pyglet.window.FPSDisplay(self)

    def _apply_camera_bounds(self) -> None:
        self.left = self.camera_x - 0.5 * self.zoomed_width
        self.right = self.camera_x + 0.5 * self.zoomed_width
        self.bottom = self.camera_y - 0.5 * self.zoomed_height
        self.top = self.camera_y + 0.5 * self.zoomed_height

    def _set_camera_center(self, x: float, y: float) -> None:
        self.camera_x = x
        self.camera_y = y
        self._apply_camera_bounds()

    def update_map(self, map_path, map_ext):
        """
        Update the map being drawn by the renderer. Converts image to a list of 3D points representing each obstacle pixel in the map.

        Args:
            map_path (str): absolute path to the map without extensions
            map_ext (str): extension for the map image file

        Returns:
            None
        """

        # load map metadata
        with open(map_path + ".yaml", "r") as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                map_resolution = map_metadata["resolution"]
                origin = map_metadata["origin"]
                origin_x = origin[0]
                origin_y = origin[1]
            except yaml.YAMLError as ex:
                raise RuntimeError(f"Failed to parse map YAML: {map_path}.yaml") from ex

        # load map image
        map_img = np.array(
            Image.open(map_path + map_ext).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        ).astype(np.float64)
        map_height = map_img.shape[0]
        map_width = map_img.shape[1]

        # convert map pixels to coordinates
        range_x = np.arange(map_width)
        range_y = np.arange(map_height)
        map_x, map_y = np.meshgrid(range_x, range_y)
        map_x = (map_x * map_resolution + origin_x).flatten()
        map_y = (map_y * map_resolution + origin_y).flatten()
        map_z = np.zeros(map_y.shape)
        map_coords = np.vstack((map_x, map_y, map_z))

        # mask and only leave the obstacle points
        map_mask = map_img == 0.0
        map_mask_flat = map_mask.flatten()
        map_points = RENDER_SCALE * map_coords[:, map_mask_flat].T
        if self.map_vertices is not None:
            self.map_vertices.delete()
        map_point_count = map_points.shape[0]
        map_positions = _flatten_vertices(map_points)
        map_colors = [183, 193, 222, 255] * map_point_count
        self.map_vertices = self.program.vertex_list(
            map_point_count,
            gl.GL_POINTS,  # pyright: ignore[reportPrivateImportUsage]
            batch=self.batch,
            position=("f", map_positions),
            colors=("Bn", map_colors),
        )
        self.map_points = map_points

    def on_resize(self, width, height):
        """
        Callback function on window resize, overrides inherited method, and updates camera values on top of the inherited on_resize() method.

        Potential improvements on current behavior: zoom/pan resets on window resize.

        Args:
            width (int): new width of window
            height (int): new height of window

        Returns:
            None
        """

        # call overrided function
        super().on_resize(width, height)

        # update camera value
        (width, height) = self.get_size()
        self.zoomed_width = self.zoom_level * width
        self.zoomed_height = self.zoom_level * height
        self._apply_camera_bounds()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """
        Callback function on mouse drag, overrides inherited method.

        Args:
            x (int): Distance in pixels from the left edge of the window.
            y (int): Distance in pixels from the bottom edge of the window.
            dx (int): Relative X position from the previous mouse position.
            dy (int): Relative Y position from the previous mouse position.
            buttons (int): Bitwise combination of the mouse buttons currently pressed.
            modifiers (int): Bitwise combination of any keyboard modifiers currently active.

        Returns:
            None
        """

        # pan camera
        self.camera_x -= dx * self.zoom_level
        self.camera_y -= dy * self.zoom_level
        self._apply_camera_bounds()

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        """
        Callback function on mouse scroll, overrides inherited method.

        Args:
            x (int): Distance in pixels from the left edge of the window.
            y (int): Distance in pixels from the bottom edge of the window.
            scroll_x (float): Amount of movement on the horizontal axis.
            scroll_y (float): Amount of movement on the vertical axis.

        Returns:
            None
        """

        # Get scale factor
        f = ZOOM_IN_FACTOR if scroll_y > 0 else ZOOM_OUT_FACTOR if scroll_y < 0 else 1

        # If zoom_level is in the proper range
        if 0.01 < self.zoom_level * f < 10:
            self.zoom_level *= f

            (width, height) = self.get_size()

            mouse_x = x / width
            mouse_y = y / height

            mouse_x_in_world = self.left + mouse_x * self.zoomed_width
            mouse_y_in_world = self.bottom + mouse_y * self.zoomed_height

            self.zoomed_width *= f
            self.zoomed_height *= f

            self.left = mouse_x_in_world - mouse_x * self.zoomed_width
            self.right = mouse_x_in_world + (1 - mouse_x) * self.zoomed_width
            self.bottom = mouse_y_in_world - mouse_y * self.zoomed_height
            self.top = mouse_y_in_world + (1 - mouse_y) * self.zoomed_height
            self.camera_x = 0.5 * (self.left + self.right)
            self.camera_y = 0.5 * (self.bottom + self.top)

    def on_close(self):
        """Callback when the window close button is clicked."""

        super().on_close()
        self.window_closed = True

    def on_draw(self):
        """
        Function when the pyglet is drawing. The function draws the batch created that includes the map points, the agent polygons, and the information text, and the fps display.

        Args:
            None

        Returns:
            None
        """

        # if map and poses doesn't exist, raise exception
        if self.map_points is None:
            raise Exception("Map not set for renderer.")
        if self.poses is None:
            raise Exception("Agent poses not updated for renderer.")

        # Initialize Projection matrix
        self.view = Mat4()
        self.projection = Mat4.orthogonal_projection(
            self.left, self.right, self.bottom, self.top, -8192, 8192
        )

        # Clear window with ClearColor
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)  # pyright: ignore[reportPrivateImportUsage]

        # Draw all batches
        self.batch.draw()
        self.fps_display.draw()

    def update_obs(self, obs):
        """
        Updates the renderer with the latest observation from the gym environment, including the agent poses, and the information text.

        Args:
            obs (dict): observation dict from the gym env

        Returns:
            None
        """

        self.ego_idx = obs["ego_idx"]
        poses_x = obs["poses_x"]
        poses_y = obs["poses_y"]
        poses_theta = obs["poses_theta"]

        num_agents = len(poses_x)
        if self.poses is None:
            for car in self.cars:
                car.delete()
            self.cars = []
            for i in range(num_agents):
                if i == self.ego_idx:
                    vertices_np = get_vertices(
                        np.array([0.0, 0.0, 0.0]), CAR_LENGTH, CAR_WIDTH
                    )
                    vertices = _flatten_vertices(vertices_np)
                    car = self.program.vertex_list_indexed(
                        4,
                        gl.GL_TRIANGLES,  # pyright: ignore[reportPrivateImportUsage]
                        CAR_TRIANGLE_INDICES,
                        batch=self.batch,
                        position=("f", vertices),
                        colors=(
                            "Bn",
                            [
                                172,
                                97,
                                185,
                                255,
                                172,
                                97,
                                185,
                                255,
                                172,
                                97,
                                185,
                                255,
                                172,
                                97,
                                185,
                                255,
                            ],
                        ),
                    )
                    self.cars.append(car)
                else:
                    vertices_np = get_vertices(
                        np.array([0.0, 0.0, 0.0]), CAR_LENGTH, CAR_WIDTH
                    )
                    vertices = _flatten_vertices(vertices_np)
                    car = self.program.vertex_list_indexed(
                        4,
                        gl.GL_TRIANGLES,  # pyright: ignore[reportPrivateImportUsage]
                        CAR_TRIANGLE_INDICES,
                        batch=self.batch,
                        position=("f", vertices),
                        colors=(
                            "Bn",
                            [
                                99,
                                52,
                                94,
                                255,
                                99,
                                52,
                                94,
                                255,
                                99,
                                52,
                                94,
                                255,
                                99,
                                52,
                                94,
                                255,
                            ],
                        ),
                    )
                    self.cars.append(car)

        poses = np.stack((poses_x, poses_y, poses_theta)).T
        self._set_camera_center(
            RENDER_SCALE * float(poses_x[self.ego_idx]),
            RENDER_SCALE * float(poses_y[self.ego_idx]),
        )
        for j in range(poses.shape[0]):
            vertices_np = RENDER_SCALE * get_vertices(
                poses[j, :], CAR_LENGTH, CAR_WIDTH
            )
            vertices = _flatten_vertices(vertices_np)
            self.cars[j].set_attribute_data("position", vertices)
        self.poses = poses

        self.score_label.text = (
            "Lap Time: {laptime:.2f}, Ego Lap Count: {count:.0f}".format(
                laptime=obs["lap_times"][0], count=obs["lap_counts"][obs["ego_idx"]]
            )
        )
