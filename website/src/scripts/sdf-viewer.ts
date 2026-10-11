// Live WebGL2 sphere tracer for the repo's GLSL SDF shapes. The shapes and
// the primitive library are the exact files the Python harness uses.
import fragTemplate from "../shaders/viewer.frag.glsl?raw";
import sdfLib from "../../../src/CAD_Harness/sdf/sdf_lib.glsl?raw";
import snowman from "../../../src/CAD_Harness/sdf/examples/snowman.glsl?raw";
import duck from "../../../src/CAD_Harness/sdf/examples/duck.glsl?raw";
import dragon from "../../../src/CAD_Harness/sdf/examples/dragon_gpt4o.glsl?raw";

export const SHAPES: Record<string, { code: string; label: string; path: string }> = {
  dragon: { code: dragon, label: "Dragon (GPT-4o)", path: "src/CAD_Harness/sdf/examples/dragon_gpt4o.glsl" },
  snowman: { code: snowman, label: "Snowman", path: "src/CAD_Harness/sdf/examples/snowman.glsl" },
  duck: { code: duck, label: "Rubber duck", path: "src/CAD_Harness/sdf/examples/duck.glsl" },
};

const VERTEX = `#version 300 es
in vec2 in_pos;
out vec2 uv;
void main() { uv = in_pos; gl_Position = vec4(in_pos, 0.0, 1.0); }`;

function fragmentFor(code: string) {
  const hasColor = /\bvec3\s+sdf_color\s*\(/.test(code) ? "#define HAS_COLOR\n" : "";
  return fragTemplate.replace("//__SDF_CODE__", `${hasColor}${sdfLib}\n${code}`);
}

function cssColor(name: string): [number, number, number] {
  const probe = document.createElement("div");
  probe.style.color = `var(${name})`;
  document.body.appendChild(probe);
  const m = getComputedStyle(probe).color.match(/[\d.]+/g) ?? ["0", "0", "0"];
  probe.remove();
  return [Number(m[0]) / 255, Number(m[1]) / 255, Number(m[2]) / 255];
}

export class SdfViewer {
  private gl: WebGL2RenderingContext;
  private prog: WebGLProgram | null = null;
  private uniforms: Record<string, WebGLUniformLocation | null> = {};
  private azimuth = 35;
  private elevation = 22;
  private distance = 3.4;
  private dirty = true;
  private visible = true;
  private bgTop: [number, number, number] = [1, 1, 1];
  private bgBottom: [number, number, number] = [1, 1, 1];
  autoRotate: boolean;
  shadows = true;
  onError: (msg: string) => void = () => {};

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, powerPreference: "high-performance" });
    if (!gl) throw new Error("WebGL2 is not available in this browser.");
    this.gl = gl;
    this.autoRotate = !matchMedia("(prefers-reduced-motion: reduce)").matches;

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

    this.bindPointer();
    new ResizeObserver(() => (this.dirty = true)).observe(canvas);
    new IntersectionObserver(([e]) => {
      this.visible = e.isIntersecting;
      this.dirty = true;
    }).observe(canvas);
    this.readTheme();
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.readTheme());
    requestAnimationFrame(this.frame);
  }

  load(code: string) {
    const gl = this.gl;
    const compile = (type: number, src: string) => {
      const sh = gl.createShader(type)!;
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh) ?? "compile error");
      return sh;
    };
    try {
      const prog = gl.createProgram()!;
      gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERTEX));
      gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fragmentFor(code)));
      gl.bindAttribLocation(prog, 0, "in_pos");
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog) ?? "link error");
      if (this.prog) gl.deleteProgram(this.prog);
      this.prog = prog;
      for (const name of ["cam_pos", "aspect", "shadows", "bg_top", "bg_bottom"]) {
        this.uniforms[name] = gl.getUniformLocation(prog, name);
      }
      this.dirty = true;
    } catch (err) {
      this.onError(String(err));
    }
  }

  private readTheme() {
    this.bgTop = cssColor("--surface-2");
    this.bgBottom = cssColor("--bg");
    this.dirty = true;
  }

  resetView() {
    this.azimuth = 35;
    this.elevation = 22;
    this.dirty = true;
  }

  private bindPointer() {
    let last: { x: number; y: number } | null = null;
    const c = this.canvas;
    c.addEventListener("pointerdown", (e) => {
      last = { x: e.clientX, y: e.clientY };
      c.setPointerCapture(e.pointerId);
      c.dataset.dragging = "true";
    });
    c.addEventListener("pointermove", (e) => {
      if (!last) return;
      this.azimuth -= (e.clientX - last.x) * 0.4;
      this.elevation = Math.min(80, Math.max(-5, this.elevation + (e.clientY - last.y) * 0.3));
      last = { x: e.clientX, y: e.clientY };
      this.dirty = true;
    });
    const end = () => {
      last = null;
      delete c.dataset.dragging;
    };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", end);
  }

  private frame = () => {
    requestAnimationFrame(this.frame);
    if (!this.prog || !this.visible) return;
    if (this.autoRotate && !this.canvas.dataset.dragging) {
      this.azimuth += 0.25;
      this.dirty = true;
    }
    if (!this.dirty) return;
    this.dirty = false;

    const gl = this.gl;
    const scale = Math.min(window.devicePixelRatio || 1, 1.5);
    const w = Math.round(this.canvas.clientWidth * scale);
    const h = Math.round(this.canvas.clientHeight * scale);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    gl.viewport(0, 0, w, h);
    gl.useProgram(this.prog);

    const az = (this.azimuth * Math.PI) / 180;
    const el = (this.elevation * Math.PI) / 180;
    const d = this.distance;
    const u = this.uniforms;
    gl.uniform3f(u.cam_pos, d * Math.cos(el) * Math.sin(az), d * Math.sin(el), d * Math.cos(el) * Math.cos(az));
    gl.uniform1f(u.aspect, w / h);
    gl.uniform1i(u.shadows, this.shadows ? 1 : 0);
    gl.uniform3f(u.bg_top, ...this.bgTop);
    gl.uniform3f(u.bg_bottom, ...this.bgBottom);

    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };

  markDirty() {
    this.dirty = true;
  }
}
