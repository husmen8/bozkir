/** The camera, in two modes.
 *
 *  Orbit matches orbit_camera() in bozkir/camera.py: z up, elevation 0
 *  along the ground, 90 straight down, eye on a sphere around a target.
 *  Fly holds the eye and turns the view. Both share azimuth and elevation,
 *  so switching keeps the view. Elevation stops just short of +-90, where
 *  forward and up are parallel and the view would spin. */
export class Orbit {
  constructor() {
    this.target = [0, 0, 0];
    this.distance = 5;
    this.azimuth = 45;
    this.elevation = 25;
    this.fov = 60;
    this.fly = false;
    this.walk = false;           // fly, held at eye height on the ground
    this.pos = [0, 0, 0];        // eye, when flying
    this.speed = 1;              // world units per second, when flying
  }

  /** Unit vector the camera looks along. */
  dir() {
    const az = this.azimuth * Math.PI / 180, el = this.elevation * Math.PI / 180;
    return [-Math.cos(el) * Math.cos(az), -Math.cos(el) * Math.sin(az),
            -Math.sin(el)];
  }

  eye() {
    if (this.fly) return this.pos.slice();
    const az = this.azimuth * Math.PI / 180, el = this.elevation * Math.PI / 180;
    return [this.target[0] + this.distance * Math.cos(el) * Math.cos(az),
            this.target[1] + this.distance * Math.cos(el) * Math.sin(az),
            this.target[2] + this.distance * Math.sin(el)];
  }

  /** Keep the view unchanged when the mode changes. */
  setFly(on) {
    if (on === this.fly) return;
    if (on) this.pos = this.eye();
    else {
      const e = this.pos, f = this.dir();
      this.target = [e[0] + f[0] * this.distance, e[1] + f[1] * this.distance,
                     e[2] + f[2] * this.distance];
    }
    this.fly = on;
  }

  basis() {
    const e = this.eye();
    const f = this.dir();
    const up = Math.abs(f[2]) > 0.999 ? [1, 0, 0] : [0, 0, 1];
    let r = [f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2],
             f[0] * up[1] - f[1] * up[0]];
    const rl = Math.hypot(...r) || 1;
    r = r.map(v => v / rl);
    const d = [f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2],
               f[0] * r[1] - f[1] * r[0]];
    return { right: r, down: d, forward: f, eye: e };
  }
}
