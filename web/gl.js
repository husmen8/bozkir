// WebGL2 plumbing: shader programs, uniform tables and textures, bound to
// one context.

/** The helpers for `gl`. */
export function glTools(gl) {
  function compile(src, type) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(s) || 'unknown error';
      const m = log.match(/ERROR:\s*\d+:(\d+)/);
      const lines = src.split('\n');
      const ctx = m ? lines.slice(Math.max(0, m[1] - 3), +m[1] + 1)
                           .map((l, i) => `${Math.max(1, m[1] - 2) + i}: ${l}`)
                           .join('\n') : '';
      throw new Error(`${type === gl.VERTEX_SHADER ? 'vertex' : 'fragment'} ` +
                      `shader failed\n${log}\n${ctx}`);
    }
    return s;
  }

  function program(vs, fs) {
    const p = gl.createProgram();
    gl.attachShader(p, compile(vs, gl.VERTEX_SHADER));
    gl.attachShader(p, compile(fs, gl.FRAGMENT_SHADER));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(p));
    }
    return p;
  }

  function uniforms(p, names) {
    const out = {};
    for (const n of names) {
      const u = 'u' + n[0].toUpperCase() + n.slice(1);
      // Array uniforms: some drivers only answer to 'name[0]'.
      out[n] = gl.getUniformLocation(p, u) || gl.getUniformLocation(p, u + '[0]');
    }
    return out;
  }

  function makeTexture(unit, internal, w, h, format, type, pixels) {
    const t = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, pixels);
    return t;
  }

  return { program, uniforms, makeTexture };
}
