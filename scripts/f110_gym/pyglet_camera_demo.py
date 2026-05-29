import pyglet
from pyglet import gl

# Zooming constants
ZOOM_IN_FACTOR = 1.2
ZOOM_OUT_FACTOR = 1 / ZOOM_IN_FACTOR


class App(pyglet.window.Window):
    def __init__(self, width, height, *args, **kwargs):
        conf = gl.Config(sample_buffers=1, samples=4, depth_size=16, double_buffer=True)
        super().__init__(width, height, config=conf, *args, **kwargs)

        # Initialize camera values
        self.left = 0
        self.right = width
        self.bottom = 0
        self.top = height
        self.zoom_level = 1
        self.zoomed_width = width
        self.zoomed_height = height

    def init_gl(self, width, height):
        # Set clear color
        gl.glClearColor(0 / 255, 0 / 255, 0 / 255, 0 / 255)

        # Set antialiasing
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glEnable(gl.GL_POLYGON_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)

        # Set alpha blending
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # Set viewport
        gl.glViewport(0, 0, width, height)

    def on_resize(self, width, height):
        super().on_resize(width, height)
        size = self.get_size()
        self.left = 0
        self.right = size[0]
        self.bottom = 0
        self.top = size[1]
        self.zoomed_width = size[0]
        self.zoomed_height = size[1]

        # # Set window values
        # self.width  = width
        # self.height = height
        # # Initialize OpenGL context
        # self.init_gl(width, height)
        # self.width = width
        # self.height = height
        # pass

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        # Move camera
        self.left -= dx * self.zoom_level
        self.right -= dx * self.zoom_level
        self.bottom -= dy * self.zoom_level
        self.top -= dy * self.zoom_level

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        # Get scale factor
        f = ZOOM_IN_FACTOR if scroll_y > 0 else ZOOM_OUT_FACTOR if scroll_y < 0 else 1
        # If zoom_level is in the proper range
        if 0.2 < self.zoom_level * f < 5:
            self.zoom_level *= f

            size = self.get_size()

            mouse_x = x / size[0]
            mouse_y = y / size[1]

            mouse_x_in_world = self.left + mouse_x * self.zoomed_width
            mouse_y_in_world = self.bottom + mouse_y * self.zoomed_height

            self.zoomed_width *= f
            self.zoomed_height *= f

            self.left = mouse_x_in_world - mouse_x * self.zoomed_width
            self.right = mouse_x_in_world + (1 - mouse_x) * self.zoomed_width
            self.bottom = mouse_y_in_world - mouse_y * self.zoomed_height
            self.top = mouse_y_in_world + (1 - mouse_y) * self.zoomed_height

    def on_draw(self):
        # Initialize Projection matrix
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()

        # Initialize Modelview matrix
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        # Save the default modelview matrix
        gl.glPushMatrix()

        # Clear window with ClearColor
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Set orthographic projection matrix
        gl.glOrtho(self.left, self.right, self.bottom, self.top, 1, -1)

        # Draw quad
        gl.glBegin(gl.GL_QUADS)
        gl.glColor3ub(0xFF, 0, 0)
        gl.glVertex2i(10, 10)

        gl.glColor3ub(0xFF, 0xFF, 0)
        gl.glVertex2i(110, 10)

        gl.glColor3ub(0, 0xFF, 0)
        gl.glVertex2i(110, 110)

        gl.glColor3ub(0, 0, 0xFF)
        gl.glVertex2i(10, 110)
        gl.glEnd()

        # Remove default modelview matrix
        gl.glPopMatrix()

    def run(self):
        pyglet.app.run()


App(800, 800, resizable=True).run()
