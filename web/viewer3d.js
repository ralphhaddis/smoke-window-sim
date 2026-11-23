/* WebGL2 3D Volumetric Raymarcher + Architectural UK Casement Room Viewer */

(function (global) {
  "use strict";

  // --- Minimal 3D Vector & Column-Major Mat4 Utilities ---
  function v3Sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function v3Add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
  function v3Scale(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
  function v3Dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function v3Cross(a, b) {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
  }
  function v3Norm(a) {
    const l = Math.hypot(a[0], a[1], a[2]) || 1.0;
    return [a[0] / l, a[1] / l, a[2] / l];
  }

  function mat4Perspective(fovyRad, aspect, near, far) {
    const f = 1.0 / Math.tan(fovyRad * 0.5);
    const nf = 1.0 / (near - far);
    const out = new Float32Array(16);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = (far + near) * nf;
    out[11] = -1.0;
    out[14] = 2.0 * far * near * nf;
    return out;
  }

  function mat4LookAt(eye, target, up) {
    const z = v3Norm(v3Sub(eye, target));
    const x = v3Norm(v3Cross(up, z));
    const y = v3Cross(z, x);
    const out = new Float32Array(16);
    out[0] = x[0]; out[1] = y[0]; out[2] = z[0]; out[3] = 0;
    out[4] = x[1]; out[5] = y[1]; out[6] = z[1]; out[7] = 0;
    out[8] = x[2]; out[9] = y[2]; out[10] = z[2]; out[11] = 0;
    out[12] = -v3Dot(x, eye);
    out[13] = -v3Dot(y, eye);
    out[14] = -v3Dot(z, eye);
    out[15] = 1;
    return out;
  }

  function mat4Multiply(a, b) {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        out[c * 4 + r] =
          a[0 * 4 + r] * b[c * 4 + 0] +
          a[1 * 4 + r] * b[c * 4 + 1] +
          a[2 * 4 + r] * b[c * 4 + 2] +
          a[3 * 4 + r] * b[c * 4 + 3];
      }
    }
    return out;
  }

  // --- Shaders ---
  const GEOM_VS = `#version 300 es
  precision highp float;
  layout(location = 0) in vec3 aPos;
  layout(location = 1) in vec4 aCol;
  uniform mat4 uVP;
  out vec4 vCol;
  void main() {
    vCol = aCol;
    gl_Position = uVP * vec4(aPos, 1.0);
  }`;

  const GEOM_FS = `#version 300 es
  precision highp float;
  in vec4 vCol;
  out vec4 fragColor;
  void main() {
    fragColor = vCol;
  }`;

  const VOL_VS = `#version 300 es
  precision highp float;
  layout(location = 0) in vec3 aUnitPos;
  uniform mat4 uVP;
  uniform vec3 uDomainSize;
  out vec3 vWorldPos;
  void main() {
    vec3 wp = aUnitPos * uDomainSize;
    vWorldPos = wp;
    gl_Position = uVP * vec4(wp, 1.0);
  }`;

  const VOL_FS = `#version 300 es
  precision highp float;
  precision highp sampler3D;

  in vec3 vWorldPos;
  out vec4 fragColor;

  uniform sampler3D uVolume;
  uniform vec3 uCamPos;
  uniform vec3 uDomainSize;
  uniform float uDensityGain;
  uniform int uMode;          // 0=smoke, 1=temp, 2=speed, 3=vorticity
  uniform float uTempMidNorm; // normalized neutral temperature in [0,1]

  vec2 intersectBox(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax) {
    vec3 invR = 1.0 / rd;
    vec3 t0s = (bmin - ro) * invR;
    vec3 t1s = (bmax - ro) * invR;
    vec3 tmin = min(t0s, t1s);
    vec3 tmax = max(t0s, t1s);
    float tNear = max(max(tmin.x, tmin.y), tmin.z);
    float tFar  = min(min(tmax.x, tmax.y), tmax.z);
    return vec2(tNear, tFar);
  }

  float ignNoise(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
  }

  vec3 viridisApprox(float t) {
    vec3 c0 = vec3(0.267, 0.005, 0.329);
    vec3 c1 = vec3(0.190, 0.407, 0.556);
    vec3 c2 = vec3(0.208, 0.718, 0.472);
    vec3 c3 = vec3(0.993, 0.906, 0.144);
    if (t < 0.333) return mix(c0, c1, t * 3.0);
    if (t < 0.666) return mix(c1, c2, (t - 0.333) * 3.0);
    return mix(c2, c3, (t - 0.666) * 3.0);
  }

  vec3 infernoApprox(float t) {
    vec3 c0 = vec3(0.08, 0.03, 0.18);
    vec3 c1 = vec3(0.57, 0.14, 0.40);
    vec3 c2 = vec3(0.93, 0.48, 0.13);
    vec3 c3 = vec3(0.99, 0.96, 0.62);
    if (t < 0.333) return mix(c0, c1, t * 3.0);
    if (t < 0.666) return mix(c1, c2, (t - 0.333) * 3.0);
    return mix(c2, c3, (t - 0.666) * 3.0);
  }

  void main() {
    vec3 rd = normalize(vWorldPos - uCamPos);
    vec2 hit = intersectBox(uCamPos, rd, vec3(0.0), uDomainSize);
    float tStart = max(hit.x, 0.0);
    float tEnd = hit.y;
    if (tStart >= tEnd) {
      discard;
    }

    float minDim = min(uDomainSize.x, min(uDomainSize.y, uDomainSize.z));
    float ds = max(minDim / 52.0, (tEnd - tStart) / 76.0);
    float jitter = ignNoise(gl_FragCoord.xy);
    float t = tStart + jitter * ds;

    vec3 invDomain = 1.0 / uDomainSize;
    vec3 sunDir = normalize(vec3(0.45, -0.30, 0.84));
    vec3 accumRgb = vec3(0.0);
    float accumA = 0.0;

    for (int step = 0; step < 76; step++) {
      if (t >= tEnd || accumA > 0.94) break;
      vec3 p = uCamPos + rd * t;
      vec3 uvw = p * invDomain;
      float v = texture(uVolume, uvw).r;

      vec3 col = vec3(0.0);
      float sigma = 0.0;

      if (uMode == 0) {
        // Physically inspired Beer-Lambert volumetric smoke with directional scattering
        float d = smoothstep(0.012, 0.92, v);
        if (d > 0.012) {
          float vLit = texture(uVolume, clamp(uvw + sunDir * 0.045, 0.0, 1.0)).r;
          float shade = exp(-2.3 * vLit * uDensityGain);
          vec3 darkCol = vec3(0.32, 0.41, 0.54);
          vec3 litCol  = vec3(0.93, 0.96, 1.00);
          col = mix(darkCol, litCol, 0.28 + 0.72 * shade);
          sigma = (0.75 * d + 2.35 * d * d) * uDensityGain;
        }
      } else if (uMode == 1) {
        // Diverging buoyancy temperature plume: blue = cold inflow, orange/red = warm outflow
        float dev = (v - uTempMidNorm) * 2.0;
        float mag = abs(dev);
        if (mag > 0.06) {
          float s = clamp((mag - 0.06) / 0.85, 0.0, 1.0);
          if (dev < 0.0) {
            col = mix(vec3(0.18, 0.65, 0.96), vec3(0.08, 0.28, 0.85), s);
          } else {
            col = mix(vec3(0.96, 0.62, 0.14), vec3(0.95, 0.22, 0.16), s);
          }
          sigma = pow(s, 1.35) * 1.45 * uDensityGain;
        }
      } else if (uMode == 2) {
        // Flow speed |u|
        float s = smoothstep(0.08, 0.95, v);
        if (s > 0.004) {
          col = viridisApprox(s);
          sigma = s * s * 1.65 * uDensityGain;
        }
      } else {
        // Vorticity |omega|
        float s = smoothstep(0.10, 0.95, v);
        if (s > 0.004) {
          col = infernoApprox(s);
          sigma = s * s * 1.85 * uDensityGain;
        }
      }

      if (sigma > 0.0005) {
        float a = 1.0 - exp(-sigma * ds);
        accumRgb += (1.0 - accumA) * col * a;
        accumA += (1.0 - accumA) * a;
      }
      t += ds;
    }

    if (accumA < 0.002) {
      discard;
    }
    fragColor = vec4(accumRgb, accumA);
  }`;

  function compileShader(gl, type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(sh);
      gl.deleteShader(sh);
      throw new Error("Shader compile error: " + log);
    }
    return sh;
  }

  function createProgram(gl, vsSrc, fsSrc) {
    const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error("Program link error: " + gl.getProgramInfoLog(prog));
    }
    return prog;
  }

  // Unit cube triangles (CCW outward normals) for volume raymarch proxy
  function makeUnitCubePositions() {
    const p = [
      // -X
      0, 0, 0,  0, 0, 1,  0, 1, 1,   0, 0, 0,  0, 1, 1,  0, 1, 0,
      // +X
      1, 0, 0,  1, 1, 0,  1, 1, 1,   1, 0, 0,  1, 1, 1,  1, 0, 1,
      // -Y
      0, 0, 0,  1, 0, 0,  1, 0, 1,   0, 0, 0,  1, 0, 1,  0, 0, 1,
      // +Y
      0, 1, 0,  0, 1, 1,  1, 1, 1,   0, 1, 0,  1, 1, 1,  1, 1, 0,
      // -Z
      0, 0, 0,  0, 1, 0,  1, 1, 0,   0, 0, 0,  1, 1, 0,  1, 0, 0,
      // +Z
      0, 0, 1,  1, 0, 1,  1, 1, 1,   0, 0, 1,  1, 1, 1,  0, 1, 1,
    ];
    return new Float32Array(p);
  }

  // Dynamic vertex accumulator for colored lines and triangles (7 floats/vert: x,y,z,r,g,b,a)
  class GeomBatch {
    constructor() {
      this.data = [];
      this._cached = null;
    }
    clear() {
      this.data.length = 0;
      this._cached = null;
    }
    vert(x, y, z, col) {
      this.data.push(x, y, z, col[0], col[1], col[2], col[3]);
    }
    line(a, b, col) {
      this.vert(a[0], a[1], a[2], col);
      this.vert(b[0], b[1], b[2], col);
    }
    tri(a, b, c, col) {
      this.vert(a[0], a[1], a[2], col);
      this.vert(b[0], b[1], b[2], col);
      this.vert(c[0], c[1], c[2], col);
    }
    quad(p0, p1, p2, p3, col) {
      this.tri(p0, p1, p2, col);
      this.tri(p0, p2, p3, col);
    }
    quadWire(p0, p1, p2, p3, col) {
      this.line(p0, p1, col);
      this.line(p1, p2, col);
      this.line(p2, p3, col);
      this.line(p3, p0, col);
    }
    boxWire(x0, y0, z0, x1, y1, z1, col) {
      const c000 = [x0, y0, z0], c100 = [x1, y0, z0], c110 = [x1, y1, z0], c010 = [x0, y1, z0];
      const c001 = [x0, y0, z1], c101 = [x1, y0, z1], c111 = [x1, y1, z1], c011 = [x0, y1, z1];
      this.quadWire(c000, c100, c110, c010, col);
      this.quadWire(c001, c101, c111, c011, col);
      this.line(c000, c001, col);
      this.line(c100, c101, col);
      this.line(c110, c111, col);
      this.line(c010, c011, col);
    }
    boxSolid(x0, y0, z0, x1, y1, z1, col) {
      const c000 = [x0, y0, z0], c100 = [x1, y0, z0], c110 = [x1, y1, z0], c010 = [x0, y1, z0];
      const c001 = [x0, y0, z1], c101 = [x1, y0, z1], c111 = [x1, y1, z1], c011 = [x0, y1, z1];
      this.quad(c000, c010, c110, c100, col); // -z
      this.quad(c001, c101, c111, c011, col); // +z
      this.quad(c000, c100, c101, c001, col); // -y
      this.quad(c010, c011, c111, c110, col); // +y
      this.quad(c000, c001, c011, c010, col); // -x
      this.quad(c100, c110, c111, c101, col); // +x
    }
    arrow(origin, dirVec, length, col) {
      const d = v3Norm(dirVec);
      const tip = v3Add(origin, v3Scale(d, length));
      this.line(origin, tip, col);
      const ref = Math.abs(d[2]) < 0.85 ? [0, 0, 1] : [0, 1, 0];
      const side1 = v3Norm(v3Cross(d, ref));
      const side2 = v3Cross(d, side1);
      const headLen = Math.min(length * 0.28, 0.22);
      const headRad = headLen * 0.45;
      const base = v3Sub(tip, v3Scale(d, headLen));
      this.line(tip, v3Add(base, v3Scale(side1, headRad)), col);
      this.line(tip, v3Sub(base, v3Scale(side1, headRad)), col);
      this.line(tip, v3Add(base, v3Scale(side2, headRad)), col);
      this.line(tip, v3Sub(base, v3Scale(side2, headRad)), col);
    }
    toFloat32() {
      if (!this._cached) {
        this._cached = new Float32Array(this.data);
      }
      return this._cached;
    }
  }

  class Viewer3D {
    constructor(canvas, fallbackEl) {
      this.canvas = canvas;
      this.fallbackEl = fallbackEl;
      this.gl = null;
      this.ready = false;

      // Camera orbit state (Z-up)
      this.target = [2.9, 2.6, 1.35];
      this.azim = -0.82;   // radians
      this.elev = 0.36;    // radians
      this.dist = 8.4;

      // Display options
      this.volMode = 0;        // 0=smoke, 1=temp, 2=speed, 3=vorticity
      this.densityGain = 1.35;
      this.tempMidNorm = 0.5;
      this.showCutPlanes = true;
      this.showMasonry = true;

      // Latest simulation state
      this.geom = null;
      this.params = null;
      this.sliceIndices = { xz_j: 32, xy_k: 18, yz_i: 44 };
      this.volDims = [0, 0, 0];
      this.hasVolume = false;

      // Reusable batches
      this.bgTris = new GeomBatch();
      this.bgLines = new GeomBatch();
      this.fgTris = new GeomBatch();
      this.fgLines = new GeomBatch();
      this.triadLines = new GeomBatch();
      this._lastGeomKey = "";

      this._initGL();
      this._initInteraction();
    }

    _initGL() {
      const gl = this.canvas.getContext("webgl2", {
        antialias: true,
        alpha: false,
        premultipliedAlpha: false,
        powerPreference: "high-performance",
      });
      if (!gl) {
        if (this.fallbackEl) this.fallbackEl.classList.remove("hidden");
        return;
      }
      this.gl = gl;

      try {
        this.geomProg = createProgram(gl, GEOM_VS, GEOM_FS);
        this.volProg = createProgram(gl, VOL_VS, VOL_FS);
      } catch (err) {
        console.error("WebGL2 shader setup failed:", err);
        if (this.fallbackEl) {
          this.fallbackEl.textContent = "WebGL2 shader error: " + err.message;
          this.fallbackEl.classList.remove("hidden");
        }
        return;
      }

      // Uniform locations
      this.uGeomVP = gl.getUniformLocation(this.geomProg, "uVP");
      this.uVolVP = gl.getUniformLocation(this.volProg, "uVP");
      this.uVolCamPos = gl.getUniformLocation(this.volProg, "uCamPos");
      this.uVolDomainSize = gl.getUniformLocation(this.volProg, "uDomainSize");
      this.uVolTex = gl.getUniformLocation(this.volProg, "uVolume");
      this.uVolDensity = gl.getUniformLocation(this.volProg, "uDensityGain");
      this.uVolMode = gl.getUniformLocation(this.volProg, "uMode");
      this.uVolTempMid = gl.getUniformLocation(this.volProg, "uTempMidNorm");

      // Unit cube VAO for volume raymarching
      this.cubeVao = gl.createVertexArray();
      gl.bindVertexArray(this.cubeVao);
      this.cubeVbo = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.cubeVbo);
      gl.bufferData(gl.ARRAY_BUFFER, makeUnitCubePositions(), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0);

      // Dynamic geometry VAO/VBO (pos3 + col4 = 28 bytes/vertex)
      this.dynVao = gl.createVertexArray();
      gl.bindVertexArray(this.dynVao);
      this.dynVbo = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.dynVbo);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 28, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 28, 12);
      gl.bindVertexArray(null);

      // 3D texture for volumetric scalar field
      this.volTex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_3D, this.volTex);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
      gl.bindTexture(gl.TEXTURE_3D, null);

      this.ready = true;
    }

    _initInteraction() {
      const c = this.canvas;
      let dragging = false;
      let mode = "orbit";
      let lastX = 0;
      let lastY = 0;

      c.addEventListener("contextmenu", (e) => e.preventDefault());

      c.addEventListener("pointerdown", (e) => {
        dragging = true;
        mode = (e.button === 2 || e.shiftKey) ? "pan" : "orbit";
        lastX = e.clientX;
        lastY = e.clientY;
        c.setPointerCapture(e.pointerId);
      });

      c.addEventListener("pointermove", (e) => {
        if (!dragging) return;
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        lastX = e.clientX;
        lastY = e.clientY;

        if (mode === "orbit") {
          this.azim -= dx * 0.0075;
          this.elev = Math.max(-1.48, Math.min(1.48, this.elev + dy * 0.0075));
        } else {
          const eye = this._getEyePos();
          const fwd = v3Norm(v3Sub(this.target, eye));
          const right = v3Norm(v3Cross(fwd, [0, 0, 1]));
          const up = v3Cross(right, fwd);
          const scale = this.dist * 0.0014;
          this.target = v3Add(
            this.target,
            v3Add(v3Scale(right, -dx * scale), v3Scale(up, dy * scale))
          );
        }
        this.render();
      });

      const endDrag = () => { dragging = false; };
      c.addEventListener("pointerup", endDrag);
      c.addEventListener("pointercancel", endDrag);

      c.addEventListener(
        "wheel",
        (e) => {
          e.preventDefault();
          const factor = Math.exp(e.deltaY * 0.0011);
          this.dist = Math.max(1.2, Math.min(24.0, this.dist * factor));
          this.render();
        },
        { passive: false }
      );
    }

    setCameraPreset(name) {
      const g = this.geom;
      const cx = g ? g.wall_x1 * 0.78 : 2.9;
      const cy = g ? g.ly * 0.5 : 2.6;
      const cz = g ? g.room_z1 * 0.52 : 1.3;
      this.target = [cx, cy, cz];

      if (name === "iso") {
        this.azim = -0.82;
        this.elev = 0.36;
        this.dist = 8.4;
      } else if (name === "outside") {
        this.azim = 0.28;
        this.elev = 0.22;
        this.dist = 7.4;
      } else if (name === "side") {
        this.azim = -Math.PI * 0.5;
        this.elev = 0.04;
        this.dist = 8.6;
      } else if (name === "top") {
        this.azim = -Math.PI * 0.5;
        this.elev = 1.46;
        this.dist = 9.0;
      }
      this.render();
    }

    setOptions(opts) {
      if (opts.volField !== undefined) {
        const map = { smoke: 0, temp: 1, speed: 2, vorticity: 3 };
        this.volMode = map[opts.volField] !== undefined ? map[opts.volField] : 0;
      }
      if (opts.densityGain !== undefined) this.densityGain = opts.densityGain;
      if (opts.showCutPlanes !== undefined) this.showCutPlanes = opts.showCutPlanes;
      if (opts.showMasonry !== undefined) this.showMasonry = opts.showMasonry;
      this._rebuildGeometry();
      this.render();
    }

    updateScene(geom, params, sliceIndices, volData, volShape, volMin, volMax) {
      const firstInit = !this.geom && geom;
      this.geom = geom;
      this.params = params;
      if (sliceIndices) {
        this.sliceIndices = sliceIndices;
      }

      if (firstInit) {
        this.setCameraPreset("iso");
      }

      if (params && volMax > volMin) {
        // Anchor transparent neutral point strictly at outdoor ambient T_out:
        // T > T_out is always warm (amber/red), T < T_out is always cool (blue).
        const tNeutral = params.temp_out_c;
        this.tempMidNorm = Math.max(0.05, Math.min(0.95, (tNeutral - volMin) / (volMax - volMin)));
      }

      if (this.ready && volData && volShape) {
        const [nz, ny, nx] = volShape;
        const gl = this.gl;
        gl.bindTexture(gl.TEXTURE_3D, this.volTex);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        if (this.volDims[0] !== nx || this.volDims[1] !== ny || this.volDims[2] !== nz) {
          gl.texImage3D(
            gl.TEXTURE_3D,
            0,
            gl.R8,
            nx,
            ny,
            nz,
            0,
            gl.RED,
            gl.UNSIGNED_BYTE,
            volData
          );
          this.volDims = [nx, ny, nz];
        } else {
          gl.texSubImage3D(
            gl.TEXTURE_3D,
            0,
            0,
            0,
            0,
            nx,
            ny,
            nz,
            gl.RED,
            gl.UNSIGNED_BYTE,
            volData
          );
        }
        gl.bindTexture(gl.TEXTURE_3D, null);
        this.hasVolume = true;
      }

      this._rebuildGeometry();
      this.render();
    }

    _rebuildGeometry() {
      const g = this.geom;
      const p = this.params;
      if (!g) return;

      const si = this.sliceIndices || {};
      const key =
        `${g.lx}|${g.ly}|${g.lz}|${g.room_x1}|${g.wall_x1}|${g.room_y0}|${g.room_y1}|${g.room_z1}|` +
        `${g.win_y0}|${g.win_y1}|${g.win_z0}|${g.win_z1}|` +
        `${this.showMasonry ? 1 : 0}|${this.showCutPlanes ? 1 : 0}|` +
        `${si.xz_j}|${si.xy_k}|${si.yz_i}|` +
        (p
          ? `${p.win_hinge_mode}|${p.win_angle_deg}|${p.vent_enabled}|` +
            `${p.smoke_enabled}|${p.smoke_x}|${p.smoke_y}|${p.smoke_z}|` +
            `${p.smoke_jet_speed}|${p.smoke_jet_azim_deg}|${p.smoke_jet_elev_deg}|` +
            `${p.wind_speed}|${p.wind_dir_deg}`
          : "");
      if (key === this._lastGeomKey) return;
      this._lastGeomKey = key;

      this.bgTris.clear();
      this.bgLines.clear();
      this.fgTris.clear();
      this.fgLines.clear();

      // 1. Ground plane grid (0.5 m spacing)
      const gridCol = [0.16, 0.22, 0.31, 0.45];
      for (let x = 0; x <= g.lx + 1e-4; x += 0.5) {
        this.bgLines.line([x, 0, 0], [x, g.ly, 0], gridCol);
      }
      for (let y = 0; y <= g.ly + 1e-4; y += 0.5) {
        this.bgLines.line([0, y, 0], [g.lx, y, 0], gridCol);
      }

      // Room floor subtle surface
      this.bgTris.quad(
        [g.room_x0, g.room_y0, 0.002],
        [g.room_x1, g.room_y0, 0.002],
        [g.room_x1, g.room_y1, 0.002],
        [g.room_x0, g.room_y1, 0.002],
        [0.14, 0.19, 0.27, 0.55]
      );

      // 2. Masonry facade wall around the window opening (4 exact blocks)
      if (this.showMasonry) {
        const wallFill = [0.36, 0.28, 0.26, 0.22];
        const wallEdge = [0.56, 0.47, 0.44, 0.55];
        const x0 = g.room_x1, x1 = g.wall_x1;
        const blocks = [
          // Below sill
          [x0, g.room_y0, g.room_z0, x1, g.room_y1, g.win_z0],
          // Above lintel
          [x0, g.room_y0, g.win_z1, x1, g.room_y1, g.room_z1],
          // Left jamb
          [x0, g.room_y0, g.win_z0, x1, g.win_y0, g.win_z1],
          // Right jamb
          [x0, g.win_y1, g.win_z0, x1, g.room_y1, g.win_z1],
        ];
        for (const b of blocks) {
          if (b[3] > b[0] && b[4] > b[1] && b[5] > b[2]) {
            this.bgTris.boxSolid(b[0], b[1], b[2], b[3], b[4], b[5], wallFill);
            this.bgLines.boxWire(b[0], b[1], b[2], b[3], b[4], b[5], wallEdge);
          }
        }
      }

      // 3. Domain outer bounding box (very subtle)
      this.bgLines.boxWire(0, 0, 0, g.lx, g.ly, g.lz, [0.22, 0.30, 0.42, 0.28]);

      // 4. Crisp Room Wireframe & Window Reveal Frame
      this.fgLines.boxWire(
        g.room_x0, g.room_y0, g.room_z0,
        g.room_x1, g.room_y1, g.room_z1,
        [0.45, 0.58, 0.74, 0.65]
      );

      // Window reveal tunnel edges (inner & outer frames)
      this.fgLines.boxWire(
        g.room_x1, g.win_y0, g.win_z0,
        g.wall_x1, g.win_y1, g.win_z1,
        [0.82, 0.90, 0.98, 0.85]
      );

      // 5. Hinged UK Casement Sash (exact 3D corners from C solver)
      if (g.sash_corners && g.sash_corners.length === 4) {
        const sc = g.sash_corners;
        this.fgTris.quad(sc[0], sc[1], sc[2], sc[3], [0.22, 0.74, 0.97, 0.22]);
        this.fgLines.quadWire(sc[0], sc[1], sc[2], sc[3], [0.35, 0.85, 1.0, 0.96]);
        // Subtle glazing reflection diagonal
        this.fgLines.line(sc[0], sc[2], [0.55, 0.90, 1.0, 0.28]);
        // Highlight hinge bar in amber
        const hingeMode = p ? p.win_hinge_mode : 0;
        const hA = sc[0];
        const hB = hingeMode === 2 ? sc[1] : sc[3];
        this.fgLines.line(hA, hB, [0.98, 0.72, 0.15, 1.0]);
      }

      // 6. Optional Rear Door / Vent on x = 0
      if (p && p.vent_enabled) {
        const vp0 = [0.004, g.vent_y0, g.vent_z0];
        const vp1 = [0.004, g.vent_y1, g.vent_z0];
        const vp2 = [0.004, g.vent_y1, g.vent_z1];
        const vp3 = [0.004, g.vent_y0, g.vent_z1];
        this.fgTris.quad(vp0, vp1, vp2, vp3, [0.13, 0.77, 0.37, 0.18]);
        this.fgLines.quadWire(vp0, vp1, vp2, vp3, [0.25, 0.90, 0.48, 0.90]);
      }

      // 7. Cut-Plane Indicators synchronized with 2D slices
      if (this.showCutPlanes && this.sliceIndices) {
        const yS = (this.sliceIndices.xz_j + 0.5) * g.h;
        const zS = (this.sliceIndices.xy_k + 0.5) * g.h;
        const xS = (this.sliceIndices.yz_i + 0.5) * g.h;

        // X-Z plane (cyan)
        this.fgLines.quadWire(
          [0, yS, 0], [g.lx, yS, 0], [g.lx, yS, g.lz], [0, yS, g.lz],
          [0.22, 0.74, 0.97, 0.34]
        );
        // X-Y plane (amber)
        this.fgLines.quadWire(
          [0, 0, zS], [g.lx, 0, zS], [g.lx, g.ly, zS], [0, g.ly, zS],
          [0.96, 0.62, 0.04, 0.34]
        );
        // Y-Z plane (violet)
        this.fgLines.quadWire(
          [xS, 0, 0], [xS, g.ly, 0], [xS, g.ly, g.lz], [xS, 0, g.lz],
          [0.65, 0.55, 0.98, 0.34]
        );
      }

      // 8. Smoke Emitter 3D Marker + Jet Arrow
      if (p && p.smoke_enabled) {
        const ex = Math.max(0.05, Math.min(g.lx - 0.05, p.smoke_x));
        const ey = Math.max(0.05, Math.min(g.ly - 0.05, g.room_y0 + p.smoke_y));
        const ez = Math.max(0.05, Math.min(g.lz - 0.05, p.smoke_z));
        const r = 0.11;
        const emCol = [0.98, 0.68, 0.14, 1.0];
        this.fgLines.line([ex - r, ey, ez], [ex + r, ey, ez], emCol);
        this.fgLines.line([ex, ey - r, ez], [ex, ey + r, ez], emCol);
        this.fgLines.line([ex, ey, ez - r], [ex, ey, ez + r], emCol);
        // Octahedron wireframe
        const px = [ex + r, ey, ez], nx = [ex - r, ey, ez];
        const py = [ex, ey + r, ez], ny = [ex, ey - r, ez];
        const pz = [ex, ey, ez + r], nz = [ex, ey, ez - r];
        this.fgLines.quadWire(px, py, nx, ny, [0.98, 0.68, 0.14, 0.75]);
        this.fgLines.quadWire(px, pz, nx, nz, [0.98, 0.68, 0.14, 0.75]);

        if (p.smoke_jet_speed > 0.02) {
          const az = (p.smoke_jet_azim_deg * Math.PI) / 180;
          const el = (p.smoke_jet_elev_deg * Math.PI) / 180;
          const dir = [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)];
          const len = 0.32 + Math.min(0.85, p.smoke_jet_speed * 0.35);
          this.fgLines.arrow([ex, ey, ez], dir, len, [1.0, 0.82, 0.28, 1.0]);
        }
      }

      // 9. Outdoor Ambient Wind Vector Indicator
      if (p && p.wind_speed > 0.05) {
        const rad = (p.wind_dir_deg * Math.PI) / 180;
        const dir = [-Math.cos(rad), Math.sin(rad), 0];
        const ox = g.wall_x1 + 0.52 * (g.lx - g.wall_x1);
        const oy = g.ly * 0.5;
        const oz = Math.min(g.lz - 0.35, g.room_z1 + 0.35);
        const len = 0.55 + Math.min(0.95, p.wind_speed * 0.18);
        const start = v3Sub([ox, oy, oz], v3Scale(dir, len * 0.5));
        this.fgLines.arrow(start, dir, len, [0.22, 0.85, 0.92, 0.92]);
      }
    }

    _getEyePos() {
      const ce = Math.cos(this.elev);
      return [
        this.target[0] + this.dist * ce * Math.cos(this.azim),
        this.target[1] + this.dist * ce * Math.sin(this.azim),
        this.target[2] + this.dist * Math.sin(this.elev),
      ];
    }

    _drawBatch(batch, mode, vp) {
      if (!batch.data.length) return;
      const gl = this.gl;
      const arr = batch.toFloat32();
      gl.useProgram(this.geomProg);
      gl.uniformMatrix4fv(this.uGeomVP, false, vp);
      gl.bindVertexArray(this.dynVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.dynVbo);
      gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STREAM_DRAW);
      gl.drawArrays(mode, 0, arr.length / 7);
      gl.bindVertexArray(null);
    }

    _renderAxisTriad(w, h) {
      const gl = this.gl;
      const size = Math.min(82, Math.floor(Math.min(w, h) * 0.24));
      gl.viewport(8, 26, size, size);
      gl.disable(gl.DEPTH_TEST);

      const ce = Math.cos(this.elev);
      const eye = [3.1 * ce * Math.cos(this.azim), 3.1 * ce * Math.sin(this.azim), 3.1 * Math.sin(this.elev)];
      const proj = mat4Perspective(0.75, 1.0, 0.1, 20.0);
      const view = mat4LookAt(eye, [0, 0, 0], [0, 0, 1]);
      const vp = mat4Multiply(proj, view);

      this.triadLines.clear();
      this.triadLines.arrow([0, 0, 0], [1, 0, 0], 1.0, [0.97, 0.44, 0.44, 0.95]);
      this.triadLines.arrow([0, 0, 0], [0, 1, 0], 1.0, [0.29, 0.87, 0.50, 0.95]);
      this.triadLines.arrow([0, 0, 0], [0, 0, 1], 1.0, [0.38, 0.65, 0.98, 0.95]);
      this._drawBatch(this.triadLines, gl.LINES, vp);
    }

    render() {
      if (!this.ready) return;
      const gl = this.gl;
      // Cap volumetric raymarch canvas to 1.0 DPR (4x fewer fragment shaders on Retina displays)
      const dpr = 1.0;
      const cw = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
      const ch = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
      if (this.canvas.width !== cw || this.canvas.height !== ch) {
        this.canvas.width = cw;
        this.canvas.height = ch;
      }

      gl.viewport(0, 0, cw, ch);
      gl.clearColor(0.055, 0.059, 0.067, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      const eye = this._getEyePos();
      const proj = mat4Perspective((42 * Math.PI) / 180, cw / ch, 0.05, 80.0);
      const view = mat4LookAt(eye, this.target, [0, 0, 1]);
      const vp = mat4Multiply(proj, view);

      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.enable(gl.DEPTH_TEST);
      gl.depthMask(false);

      // Pass 1: Ground grid, floor & translucent masonry wall slabs
      this._drawBatch(this.bgLines, gl.LINES, vp);
      this._drawBatch(this.bgTris, gl.TRIANGLES, vp);

      // Pass 2: WebGL2 3D Volumetric Raymarching
      if (this.hasVolume && this.geom) {
        gl.disable(gl.DEPTH_TEST);
        gl.enable(gl.CULL_FACE);
        gl.cullFace(gl.FRONT);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

        gl.useProgram(this.volProg);
        gl.uniformMatrix4fv(this.uVolVP, false, vp);
        gl.uniform3f(this.uVolCamPos, eye[0], eye[1], eye[2]);
        gl.uniform3f(this.uVolDomainSize, this.geom.lx, this.geom.ly, this.geom.lz);
        gl.uniform1f(this.uVolDensity, this.densityGain);
        gl.uniform1i(this.uVolMode, this.volMode);
        gl.uniform1f(this.uVolTempMid, this.tempMidNorm);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_3D, this.volTex);
        gl.uniform1i(this.uVolTex, 0);

        gl.bindVertexArray(this.cubeVao);
        gl.drawArrays(gl.TRIANGLES, 0, 36);
        gl.bindVertexArray(null);
        gl.disable(gl.CULL_FACE);
      }

      // Pass 3: Foreground architectural wireframe, casement sash glass, emitter, cut-planes
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.disable(gl.DEPTH_TEST);
      this._drawBatch(this.fgTris, gl.TRIANGLES, vp);
      this._drawBatch(this.fgLines, gl.LINES, vp);

      // Pass 4: Corner 3D Axis Triad
      this._renderAxisTriad(cw, ch);
      gl.depthMask(true);
    }
  }

  global.Viewer3D = Viewer3D;
})(window);
