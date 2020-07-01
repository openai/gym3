"""
Class for rendering bitmaps and text in a window with OpenGL
"""

import os
from typing import Any, Optional, Set, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT: Optional[np.ndarray] = None


def _str_to_array(s: str) -> np.ndarray:
    """
    Convert a text string into a numpy array
    """
    max_line_length = max(len(line) for line in s.split("\n"))

    lines = []
    for line in s.split("\n"):
        lines.append(line + " " * (max_line_length - len(line)))

    arrs = []
    for line in lines:
        arr = np.frombuffer(line.encode("utf8"), dtype=np.uint8)
        arrs.append(arr)
    return np.stack(arrs)


def _convert_ascii_to_rgba(arr: np.ndarray, size_px=32) -> np.ndarray:
    """
    Convert an ascii array to an image array using the loaded font
    """
    global FONT
    if FONT is None:
        FONT = np.load(os.path.join(SCRIPT_DIR, "font.bin"))

    charset = FONT[f"{size_px}px"]
    _, char_height, char_width = charset.shape
    height, width = arr.shape
    image = np.ones((height * char_height, width * char_width, 4), dtype=np.uint8) * 255
    for y in range(height):
        for x in range(width):
            ch = arr[y, x]
            image[
                y * char_height : (y + 1) * char_height,
                x * char_width : (x + 1) * char_width,
                3,
            ] = charset[ch]
    return image


BITMAP_VERTEX_SHADER = """
#version 330

uniform float in_width;
uniform float in_height;
in vec2 in_pos;
in vec2 in_tex_coord;
out vec2 tex_coord;

void main() {
    // convert from screen pixels to normalized device coordinates
    gl_Position = vec4(in_pos.x / (in_width / 2) - 1, in_pos.y / (in_height / 2) - 1, 0.0, 1.0);
    tex_coord = in_tex_coord;
}
"""

BITMAP_FRAGMENT_SHADER = """
#version 330

in vec2 tex_coord;
uniform float in_alpha;
uniform sampler2D sampler;
out vec4 out_frag_color;

void main() {
    out_frag_color = texture(sampler, tex_coord);
    // set in_alpha to -1 to use the original alpha of the texture
    if (in_alpha != -1.0) {
        out_frag_color.a = in_alpha;
    }
}
"""


class Renderer:
    """
    A simple window for rendering that uses OpenGL to display bitmaps or text
    and returns key presses.

    Subclasses can override behavior to provide environment-specific drawing.
    """

    def __init__(self, width: int, height: int) -> None:
        # do late imports of these to avoid any interference with other opengl libraries
        try:
            import glfw

            self._glfw = glfw
        except ImportError as e:
            raise Exception(
                f"failed to import glfw: '{e}', please make sure you have the newest version of glfw with `pip install --upgrade glfw`"
            )
        import moderngl

        self._mgl = moderngl

        if not self._glfw.init():
            raise Exception("failed to initialize glfw")

        self._glfw.window_hint(self._glfw.CLIENT_API, self._glfw.OPENGL_API)
        self._glfw.window_hint(self._glfw.CONTEXT_VERSION_MAJOR, 3)
        self._glfw.window_hint(self._glfw.CONTEXT_VERSION_MINOR, 3)
        self._glfw.window_hint(
            self._glfw.OPENGL_PROFILE, self._glfw.OPENGL_CORE_PROFILE
        )
        self._glfw.window_hint(self._glfw.OPENGL_FORWARD_COMPAT, True)
        self._glfw.window_hint(self._glfw.RESIZABLE, False)
        self._glfw.window_hint(self._glfw.DOUBLEBUFFER, True)
        self._glfw.window_hint(self._glfw.DEPTH_BITS, 24)

        self.width = width
        self.height = height
        self.is_open = True
        self._should_close = False

        self._window = self._glfw.create_window(
            self.width, self.height, "Gym3 Viewer", None, None
        )
        if not self._window:
            self._glfw.terminate()
            raise Exception("failed to create window")

        # self._glfw.get_key_name doesn't handle non-text keys
        self._key_to_name = {
            getattr(self._glfw, attr): attr.split("_", 1)[1]
            for attr in dir(self._glfw)
            if attr.startswith("KEY_")
        }
        self._keys_clicked = set()
        self._keys_pressed = set()
        self._glfw.set_key_callback(self._window, self._on_key_event)

        self._glfw.make_context_current(self._window)

        self._ctx = self._mgl.create_context()
        self._ctx.enable_only(self._mgl.BLEND)
        self._ctx.blend_func = self._mgl.SRC_ALPHA, self._mgl.ONE_MINUS_SRC_ALPHA

        self._bitmap_shader = self._ctx.program(
            vertex_shader=BITMAP_VERTEX_SHADER, fragment_shader=BITMAP_FRAGMENT_SHADER
        )

        self._bitmap_shader["in_width"].value = self.width
        self._bitmap_shader["in_height"].value = self.height

        self._vbo = None
        self._vao = None

    def _on_key_event(
        self, window: Any, key: int, scancode: int, action: int, mode: int
    ) -> None:
        name = self._key_to_name.get(key)
        if action == self._glfw.PRESS:
            self._keys_pressed.add(name)
            self._keys_clicked.add(name)
        elif action == self._glfw.RELEASE:
            if name in self._keys_pressed:
                # hitting "fn" on a mac only seems to produce the RELEASE action
                self._keys_pressed.remove(name)

    def get_time(self) -> float:
        """
        Get an accurate time using glfw.get_time()
        """
        return self._glfw.get_time()

    def start(self) -> Tuple[Set, Set]:
        """
        Start a new frame

        Returns:
            keys_clicked: keys the user has pressed since the last time finish() was called
            keys_pressed: keys the user currently has held down
        """
        self._glfw.poll_events()
        self._ctx.screen.clear(0.0, 0.0, 0.0, 1.0)
        self._should_close = "ESCAPE" in self._keys_clicked
        keys_clicked = self._keys_clicked
        self._keys_clicked = set()
        return keys_clicked, self._keys_pressed

    def draw_bitmap(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        image: np.ndarray,
        antialias: bool = False,
        alpha: float = 1.0,
    ) -> None:
        """
        Draw a bitmap to the screen at the location (x, y) with size (w, h) all units are in screen pixels

        Args:
            x: x position relative to left side of screen in pixels
            y: y position relative to bottom side of screen in pixels
            w: width of image in pixels
            h: height of image in pixels
            image: a numpy array of the image to draw
            antialias: if set to True, use antialiasing then drawing the bitmap
            alpha: how opaque to make the bitmap, 1.0 is fully opaque, 0.0 is transparent, -1.0 means use
                the alpha channel of the image
        """
        tex = self._ctx.texture(
            size=(image.shape[1], image.shape[0]),
            components=image.shape[2],
            data=image.tobytes(),
        )
        if not antialias:
            tex.filter = (self._mgl.NEAREST, self._mgl.NEAREST)
        tex.use(location=self._bitmap_shader["sampler"].value)
        self._bitmap_shader["in_alpha"].value = alpha
        # textures are expected to start with the last row of the image (the lower left corner)
        # but numpy format starts with the first row (top left corner)
        # as a result, we need to flip the t values in our texture coordinates
        vertices = np.array(
            [  # x, y, s, t
                [x, y, 0, 1],
                [x + w, y, 1, 1],
                [x, y + h, 0, 0],
                [x + w, y + h, 1, 0],
            ],
            dtype=np.float32,
        )
        if self._vbo is None:
            self._vbo = self._ctx.buffer(vertices.tobytes())
        else:
            self._vbo.write(vertices.tobytes())

        if self._vao is None:
            self._vao = self._ctx.simple_vertex_array(
                self._bitmap_shader, self._vbo, "in_pos", "in_tex_coord"
            )

        self._glfw.make_context_current(self._window)
        self._vao.render(self._mgl.TRIANGLE_STRIP)
        tex.release()

    def draw_text(
        self,
        x: float,
        y: float,
        text: str,
        size_px: int = 32,
        centered: bool = False,
        bg_alpha: float = 0.0,
    ) -> None:
        """
        Draw a multi-line text string `text` to the screen at the indicated location (x, y)

        Args:
            x: x position relative to left side of screen in pixels
            y: y position relative to bottom side of screen in pixels
            text: text to draw, multiple lines are fine
            size_px: what size font to use in pixels
            centered: if set to True, x and y specify the center of the resulting text box rather
                than the bottom left corner
            bg_alpha: opacity of black background that is drawn automatically behind the text
        """
        arr = _str_to_array(text)
        image = _convert_ascii_to_rgba(arr, size_px=size_px)
        w = image.shape[1]
        h = image.shape[0]
        if centered:
            x -= w / 2
            y -= h / 2
        self.draw_bitmap(
            x, y, w, h, image=np.zeros((1, 1, 3), dtype=np.uint8), alpha=bg_alpha
        )
        self.draw_bitmap(x=x, y=y, w=w, h=h, image=image, alpha=-1.0)

    def finish(self) -> None:
        """
        Complete the current frame and return keyboard keys that the user has input
        """
        self._glfw.swap_buffers(self._window)

        if self._should_close or self._glfw.window_should_close(self._window):
            self._glfw.destroy_window(self._window)
            self.is_open = False


def main():
    r = Renderer(width=768, height=768)
    for i in range(1000):
        r.start()
        r.draw_text(i, r.height // 2, "meow!")
        r.finish()


if __name__ == "__main__":
    main()
