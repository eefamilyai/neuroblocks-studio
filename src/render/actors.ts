/**
 * Visuals for everything that moves: the robot, objects (coins, balls,
 * bots…), the "ghost" swarm of training robots, ray-vision lines and trails.
 */
import * as THREE from 'three';
import type { Env, ObjState } from '../sim/env.ts';
import { ALL_TAGS } from '../sim/env.ts';
import { TAG_INFO } from '../sim/types.ts';

const tmpM = new THREE.Matrix4();
const tmpQ = new THREE.Quaternion();
const tmpV = new THREE.Vector3();
const tmpS = new THREE.Vector3(1, 1, 1);
const UP = new THREE.Vector3(0, 1, 0);

export class RobotMesh extends THREE.Group {
  private bodyMat: THREE.MeshStandardMaterial;
  private eyeMat: THREE.MeshStandardMaterial;
  private tipMat: THREE.MeshStandardMaterial;
  private body: THREE.Group;
  private eyes: THREE.Group;
  private mood = 0;
  private lean = new THREE.Vector2();
  private lastPos = new THREE.Vector3();
  private lastVel = new THREE.Vector3();
  thrusters: THREE.Mesh;

  constructor(color: string) {
    super();
    this.body = new THREE.Group();
    this.add(this.body);
    this.bodyMat = new THREE.MeshStandardMaterial({ color, roughness: 0.3, metalness: 0.15 });
    const shell = new THREE.Mesh(new THREE.SphereGeometry(0.5, 32, 20), this.bodyMat);
    shell.scale.set(1, 0.92, 1);
    shell.castShadow = true;
    this.body.add(shell);
    const band = new THREE.Mesh(new THREE.TorusGeometry(0.47, 0.045, 8, 40).rotateX(Math.PI / 2), new THREE.MeshStandardMaterial({ color: 0xf2f4ff, roughness: 0.4 }));
    band.position.y = -0.08;
    this.body.add(band);
    const visor = new THREE.Mesh(
      new THREE.SphereGeometry(0.43, 28, 16, -Math.PI * 0.42, Math.PI * 0.84, Math.PI * 0.22, Math.PI * 0.36),
      new THREE.MeshStandardMaterial({ color: 0x0b0f1f, roughness: 0.15, metalness: 0.6 }),
    );
    visor.position.z = 0.1;
    visor.rotation.y = 0;
    this.body.add(visor);
    this.eyeMat = new THREE.MeshStandardMaterial({ color: 0x7ff6ff, emissive: 0x4fe3ff, emissiveIntensity: 2.2 });
    this.eyes = new THREE.Group();
    for (const sx of [-1, 1]) {
      const eye = new THREE.Mesh(new THREE.CapsuleGeometry(0.045, 0.07, 4, 8), this.eyeMat);
      eye.position.set(sx * 0.13, 0.12, 0.49);
      this.eyes.add(eye);
    }
    this.body.add(this.eyes);
    const ant = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 0.28, 6), new THREE.MeshStandardMaterial({ color: 0xc8cde0, metalness: 0.6 }));
    ant.position.set(0, 0.56, -0.05);
    this.body.add(ant);
    this.tipMat = new THREE.MeshStandardMaterial({ color: 0xffd166, emissive: 0xffc233, emissiveIntensity: 1.5 });
    const tip = new THREE.Mesh(new THREE.SphereGeometry(0.065, 12, 8), this.tipMat);
    tip.position.set(0, 0.72, -0.05);
    this.body.add(tip);
    this.thrusters = new THREE.Mesh(new THREE.TorusGeometry(0.26, 0.05, 8, 24).rotateX(Math.PI / 2), new THREE.MeshStandardMaterial({ color: 0x9be7ff, emissive: 0x39c6ff, emissiveIntensity: 1.6 }));
    this.thrusters.position.y = -0.47;
    this.add(this.thrusters);
  }

  setColor(c: string): void {
    this.bodyMat.color.set(c);
  }

  /** Flash the eyes: >0 happy (green), <0 sad (red). */
  react(r: number): void {
    if (Math.abs(r) > 0.04) this.mood = Math.max(-1, Math.min(1, r > 0 ? 1 : -1));
  }

  animate(dt: number, t: number, yaw: number, vx: number, vz: number, turn: number): void {
    this.rotation.y = yaw;
    // lean into acceleration, like a little hover-bot
    const ax = (vx - this.lastVel.x) / Math.max(dt, 1e-3);
    const az = (vz - this.lastVel.z) / Math.max(dt, 1e-3);
    this.lastVel.set(vx, 0, vz);
    const s = Math.sin(yaw);
    const c = Math.cos(yaw);
    const fwdA = ax * s + az * c;
    this.lean.x += (Math.max(-0.35, Math.min(0.35, fwdA * 0.03)) - this.lean.x) * Math.min(1, dt * 6);
    this.lean.y += (Math.max(-0.3, Math.min(0.3, -turn * 0.25)) - this.lean.y) * Math.min(1, dt * 6);
    this.body.rotation.x = this.lean.x;
    this.body.rotation.z = this.lean.y;
    this.body.position.y = Math.sin(t * 3.2) * 0.025;
    this.eyes.position.x = -turn * 0.04;
    // blink
    const blink = Math.sin(t * 0.9) > 0.985 ? 0.15 : 1;
    this.eyes.scale.y = blink;
    this.mood *= Math.pow(0.12, dt);
    const m = this.mood;
    if (m > 0.05) this.eyeMat.emissive.setRGB(0.3 + 0.4 * (1 - m), 1, 0.5 * (1 - m));
    else if (m < -0.05) this.eyeMat.emissive.setRGB(1, 0.35 * (1 + m), 0.35 * (1 + m));
    else this.eyeMat.emissive.setHex(0x4fe3ff);
    this.tipMat.emissiveIntensity = 1 + Math.abs(m) * 3;
    (this.thrusters.material as THREE.MeshStandardMaterial).emissiveIntensity = 1.2 + Math.sin(t * 12) * 0.3;
  }

  dispose(): void {
    this.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (mesh.geometry) mesh.geometry.dispose();
    });
  }
}

function canvasTex(draw: (g: CanvasRenderingContext2D, w: number, h: number) => void, w = 128, h = 128): THREE.CanvasTexture {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  draw(c.getContext('2d')!, w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

let glowTex: THREE.CanvasTexture | null = null;
function getGlowTex(): THREE.CanvasTexture {
  if (glowTex) return glowTex;
  glowTex = canvasTex((g, w, h) => {
    const grd = g.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w / 2);
    grd.addColorStop(0, 'rgba(255,255,255,1)');
    grd.addColorStop(0.25, 'rgba(255,255,255,0.55)');
    grd.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grd;
    g.fillRect(0, 0, w, h);
  }, 64, 64);
  return glowTex;
}

function glow(color: number, size: number): THREE.Sprite {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: getGlowTex(), color, transparent: true, opacity: 0.75, blending: THREE.AdditiveBlending, depthWrite: false }));
  s.scale.setScalar(size);
  return s;
}

let crateTex: THREE.CanvasTexture | null = null;
function getCrateTex(): THREE.CanvasTexture {
  if (crateTex) return crateTex;
  crateTex = canvasTex((g, w, h) => {
    g.fillStyle = '#c8893f';
    g.fillRect(0, 0, w, h);
    g.strokeStyle = '#8a5a26';
    g.lineWidth = 10;
    g.strokeRect(5, 5, w - 10, h - 10);
    g.beginPath();
    g.moveTo(8, 8);
    g.lineTo(w - 8, h - 8);
    g.moveTo(w - 8, 8);
    g.lineTo(8, h - 8);
    g.stroke();
    g.strokeStyle = 'rgba(0,0,0,0.12)';
    g.lineWidth = 2;
    for (let y = 20; y < h; y += 22) {
      g.beginPath();
      g.moveTo(0, y);
      g.lineTo(w, y);
      g.stroke();
    }
  });
  return crateTex;
}

let ballTex: THREE.CanvasTexture | null = null;
function getBallTex(): THREE.CanvasTexture {
  if (ballTex) return ballTex;
  ballTex = canvasTex(
    (g, w, h) => {
      g.fillStyle = '#f7f7f7';
      g.fillRect(0, 0, w, h);
      g.fillStyle = '#1b1b1b';
      const spots = [
        [0.1, 0.5],
        [0.35, 0.2],
        [0.35, 0.8],
        [0.6, 0.5],
        [0.85, 0.2],
        [0.85, 0.8],
        [0.6, 0.02],
        [0.1, 0.02],
        [0.6, 0.98],
        [0.1, 0.98],
      ];
      for (const [x, y] of spots) {
        g.beginPath();
        for (let k = 0; k < 5; k++) {
          const a = (k / 5) * Math.PI * 2;
          g.lineTo(x * w + Math.cos(a) * 16, y * h + Math.sin(a) * 26);
        }
        g.fill();
      }
    },
    256,
    128,
  );
  return ballTex;
}

/** Builds a mesh for one object type. */
function makeObjMesh(o: ObjState): THREE.Object3D {
  const g = new THREE.Group();
  switch (o.type) {
    case 'coin': {
      const coin = new THREE.Mesh(
        new THREE.CylinderGeometry(0.42, 0.42, 0.1, 24).rotateX(Math.PI / 2),
        new THREE.MeshStandardMaterial({ color: 0xffcc33, metalness: 0.85, roughness: 0.25, emissive: 0x8a5a00, emissiveIntensity: 0.6 }),
      );
      coin.position.y = 0.8;
      coin.castShadow = true;
      const inner = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.28, 0.12, 5).rotateX(Math.PI / 2), new THREE.MeshStandardMaterial({ color: 0xffe082, metalness: 0.7, roughness: 0.3 }));
      coin.add(inner);
      const halo = glow(0xffc61a, 1.9);
      halo.position.y = 0.8;
      g.add(coin, halo);
      g.userData.spin = coin;
      g.userData.halo = halo;
      break;
    }
    case 'flag': {
      const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.06, 3.2, 8), new THREE.MeshStandardMaterial({ color: 0xeeeeee, metalness: 0.6, roughness: 0.3 }));
      pole.position.y = 1.6;
      pole.castShadow = true;
      const clothGeo = new THREE.PlaneGeometry(1.3, 0.8, 12, 4);
      clothGeo.translate(0.65, 0, 0);
      const cloth = new THREE.Mesh(clothGeo, new THREE.MeshStandardMaterial({ color: 0xff3d5a, side: THREE.DoubleSide, roughness: 0.7 }));
      cloth.position.set(0.05, 2.75, 0);
      cloth.castShadow = true;
      const base = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.45, 0.15, 16), new THREE.MeshStandardMaterial({ color: 0x555a70 }));
      base.position.y = 0.075;
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.9, 1.05, 40).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: 0xff3d5a, transparent: true, opacity: 0.6 }));
      ring.position.y = 0.04;
      g.add(pole, cloth, base, ring);
      g.userData.cloth = cloth;
      g.userData.ring = ring;
      break;
    }
    case 'trophy': {
      const gold = new THREE.MeshStandardMaterial({ color: 0xffd700, metalness: 0.9, roughness: 0.2, emissive: 0x6b4d00, emissiveIntensity: 0.5 });
      const pts = [new THREE.Vector2(0.0, 0), new THREE.Vector2(0.3, 0), new THREE.Vector2(0.3, 0.08), new THREE.Vector2(0.08, 0.15), new THREE.Vector2(0.07, 0.5), new THREE.Vector2(0.4, 0.75), new THREE.Vector2(0.45, 1.25), new THREE.Vector2(0.4, 1.25), new THREE.Vector2(0.0, 0.85)];
      const cup = new THREE.Mesh(new THREE.LatheGeometry(pts, 24), gold);
      cup.position.y = 0.4;
      cup.castShadow = true;
      const ped = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.4, 0.9), new THREE.MeshStandardMaterial({ color: 0x3d4466 }));
      ped.position.y = 0.2;
      const halo = glow(0xffe066, 3);
      halo.position.y = 1.1;
      g.add(ped, cup, halo);
      g.userData.spin = cup;
      break;
    }
    case 'ball': {
      const ball = new THREE.Mesh(new THREE.SphereGeometry(o.r, 24, 16), new THREE.MeshStandardMaterial({ map: getBallTex(), roughness: 0.5 }));
      ball.castShadow = true;
      g.add(ball);
      g.userData.roll = ball;
      break;
    }
    case 'crate': {
      const box = new THREE.Mesh(new THREE.BoxGeometry(o.hx * 2, o.hy * 2, o.hz * 2), new THREE.MeshStandardMaterial({ map: getCrateTex(), roughness: 0.85 }));
      box.castShadow = true;
      box.receiveShadow = true;
      g.add(box);
      break;
    }
    case 'pad': {
      const pad = new THREE.Mesh(new THREE.BoxGeometry(o.hx * 2, 0.06, o.hz * 2), new THREE.MeshStandardMaterial({ color: 0x1fd17a, emissive: 0x0f8f4f, emissiveIntensity: 0.8, transparent: true, opacity: 0.85 }));
      pad.position.y = -o.hy + 0.53;
      const frame = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(o.hx * 2, 0.08, o.hz * 2)), new THREE.LineBasicMaterial({ color: 0xb7ffd8 }));
      frame.position.y = pad.position.y;
      g.add(pad, frame);
      g.userData.glow = pad;
      break;
    }
    case 'goal': {
      const plane = new THREE.Mesh(new THREE.BoxGeometry(o.hx * 2, o.hy * 2, o.hz * 2), new THREE.MeshBasicMaterial({ color: 0x66ddff, transparent: true, opacity: 0.08, depthWrite: false }));
      g.add(plane);
      g.userData.glow = plane;
      break;
    }
    case 'checkpoint': {
      const mat = new THREE.MeshStandardMaterial({ color: 0xa77bff, emissive: 0x6b3dff, emissiveIntensity: 1, transparent: true, opacity: 0.9 });
      const postGeo = new THREE.CylinderGeometry(0.12, 0.12, o.hy * 2, 8);
      for (const sz of [-1, 1]) {
        const post = new THREE.Mesh(postGeo, mat);
        post.position.set(0, 0, sz * o.hz);
        g.add(post);
      }
      const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, o.hz * 2, 8).rotateX(Math.PI / 2), mat);
      bar.position.y = o.hy;
      g.add(bar);
      const sheet = new THREE.Mesh(new THREE.PlaneGeometry(o.hz * 2, o.hy * 2).rotateY(Math.PI / 2), new THREE.MeshBasicMaterial({ color: 0xa77bff, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false }));
      g.add(sheet);
      g.userData.mat = mat;
      g.userData.sheet = sheet;
      break;
    }
    case 'runner': {
      const pink = new THREE.MeshStandardMaterial({ color: 0xff8fd1, roughness: 0.4 });
      const b = new THREE.Mesh(new THREE.SphereGeometry(o.r, 20, 14), pink);
      b.castShadow = true;
      const earGeo = new THREE.CapsuleGeometry(0.09, 0.45, 4, 8);
      for (const sx of [-1, 1]) {
        const ear = new THREE.Mesh(earGeo, pink);
        ear.position.set(sx * 0.17, 0.62, -0.05);
        ear.rotation.z = sx * -0.25;
        b.add(ear);
      }
      const eyeMat = new THREE.MeshBasicMaterial({ color: 0x111111 });
      for (const sx of [-1, 1]) {
        const eye = new THREE.Mesh(new THREE.SphereGeometry(0.06, 8, 6), eyeMat);
        eye.position.set(sx * 0.15, 0.12, o.r - 0.04);
        b.add(eye);
      }
      g.add(b);
      g.userData.hop = b;
      break;
    }
    case 'chaser': {
      const red = new THREE.MeshStandardMaterial({ color: 0xff3b3b, roughness: 0.35, emissive: 0x550000, emissiveIntensity: 0.6 });
      const b = new THREE.Mesh(new THREE.IcosahedronGeometry(o.r, 0), red);
      b.castShadow = true;
      const spikeGeo = new THREE.ConeGeometry(0.1, 0.32, 6);
      const ico = new THREE.IcosahedronGeometry(1, 0);
      const pos = ico.attributes.position;
      const seen = new Set<string>();
      for (let i = 0; i < pos.count; i++) {
        tmpV.fromBufferAttribute(pos, i).normalize();
        const key = tmpV.toArray().map((v) => v.toFixed(2)).join();
        if (seen.has(key)) continue;
        seen.add(key);
        const sp = new THREE.Mesh(spikeGeo, red);
        sp.position.copy(tmpV).multiplyScalar(o.r * 0.95);
        sp.quaternion.setFromUnitVectors(UP, tmpV);
        b.add(sp);
      }
      const eyeMat = new THREE.MeshBasicMaterial({ color: 0xffff66 });
      for (const sx of [-1, 1]) {
        const eye = new THREE.Mesh(new THREE.SphereGeometry(0.07, 8, 6), eyeMat);
        eye.position.set(sx * 0.16, 0.1, o.r - 0.02);
        b.add(eye);
      }
      g.add(b);
      g.userData.spin2 = b;
      break;
    }
  }
  return g;
}

interface ObjVis {
  obj: THREE.Object3D;
  type: string;
  px: number;
  py: number;
  pz: number;
  pyaw: number;
}

/** Keeps the object meshes in sync with an Env (with interpolation between physics steps). */
export class ObjectLayer {
  group = new THREE.Group();
  private vis: ObjVis[] = [];
  private env: Env | null = null;
  private prev: Float32Array = new Float32Array(0);

  bind(env: Env): void {
    for (const v of this.vis) this.group.remove(v.obj);
    this.vis = [];
    this.env = env;
    for (const o of env.objs) {
      const m = makeObjMesh(o);
      this.group.add(m);
      this.vis.push({ obj: m, type: o.type, px: o.x, py: o.y, pz: o.z, pyaw: o.yaw });
    }
    this.prev = new Float32Array(env.objs.length * 4);
    this.snapshotPrev();
  }

  /** Remember positions before a physics step (for smooth interpolation). */
  snapshotPrev(): void {
    if (!this.env) return;
    this.env.objs.forEach((o, i) => {
      this.prev[i * 4] = o.x;
      this.prev[i * 4 + 1] = o.y;
      this.prev[i * 4 + 2] = o.z;
      this.prev[i * 4 + 3] = o.yaw;
    });
  }

  update(alpha: number, t: number, dt: number): void {
    const env = this.env;
    if (!env) return;
    env.objs.forEach((o, i) => {
      const v = this.vis[i];
      const m = v.obj;
      m.visible = o.active;
      if (!o.active) return;
      let x = o.x;
      let y = o.y;
      let z = o.z;
      const px = this.prev[i * 4];
      const py = this.prev[i * 4 + 1];
      const pz = this.prev[i * 4 + 2];
      if (Math.abs(x - px) + Math.abs(z - pz) < 3) {
        x = px + (x - px) * alpha;
        y = py + (y - py) * alpha;
        z = pz + (z - pz) * alpha;
      }
      m.position.set(x, y, z);
      if (o.type === 'checkpoint' || o.type === 'goal' || o.type === 'pad') m.rotation.y = o.yaw;
      if (o.type === 'runner' || o.type === 'chaser') m.rotation.y = o.yaw;
      const ud = m.userData;
      if (ud.spin) {
        (ud.spin as THREE.Object3D).rotation.y = t * 2.2 + i;
        if (o.type === 'coin') {
          const yy = 0.8 + Math.sin(t * 2.5 + i) * 0.12;
          (ud.spin as THREE.Object3D).position.y = yy;
          if (ud.halo) (ud.halo as THREE.Object3D).position.y = yy;
        }
      }
      if (ud.cloth) {
        const geo = (ud.cloth as THREE.Mesh).geometry as THREE.PlaneGeometry;
        const p = geo.attributes.position;
        for (let k = 0; k < p.count; k++) {
          const cx = p.getX(k);
          p.setZ(k, Math.sin(cx * 4 - t * 6 + i) * 0.08 * cx);
        }
        p.needsUpdate = true;
        (ud.ring as THREE.Mesh).scale.setScalar(1 + Math.sin(t * 3) * 0.06);
      }
      if (ud.roll) {
        const ball = ud.roll as THREE.Mesh;
        const vx = (o.x - px) / Math.max(dt, 1e-3);
        const vz = (o.z - pz) / Math.max(dt, 1e-3);
        const sp = Math.hypot(vx, vz);
        if (sp > 0.01) {
          tmpV.set(vz, 0, -vx).normalize();
          tmpQ.setFromAxisAngle(tmpV, (sp * dt) / o.r);
          ball.quaternion.premultiply(tmpQ);
        }
      }
      if (ud.hop) (ud.hop as THREE.Object3D).position.y = Math.abs(Math.sin(t * 9 + i)) * 0.15;
      if (ud.spin2) (ud.spin2 as THREE.Object3D).rotation.x = t * 4;
      if (ud.mat) {
        const isNext = o.index === env.nextCp;
        const mat = ud.mat as THREE.MeshStandardMaterial;
        mat.emissiveIntensity = isNext ? 1.6 + Math.sin(t * 5) * 0.5 : 0.25;
        mat.opacity = isNext ? 1 : 0.35;
        (ud.sheet as THREE.Mesh).visible = isNext;
      }
      if (ud.glow) {
        const mat = (ud.glow as THREE.Mesh).material as THREE.MeshStandardMaterial;
        if ('emissiveIntensity' in mat) mat.emissiveIntensity = 0.6 + Math.sin(t * 3 + i) * 0.3 + (o.flash > 0 ? 2 : 0);
      }
    });
  }
}

/** The training swarm: one translucent robot per parallel world. */
export class GhostLayer {
  mesh: THREE.InstancedMesh;
  private count = 0;
  constructor(max = 128) {
    const geo = new THREE.SphereGeometry(0.46, 14, 10);
    const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, transparent: true, opacity: 0.42, roughness: 0.4, emissive: 0x223355, emissiveIntensity: 0.5, depthWrite: false });
    this.mesh = new THREE.InstancedMesh(geo, mat, max);
    this.mesh.count = 0;
    this.mesh.frustumCulled = false;
    const c = new THREE.Color();
    for (let i = 0; i < max; i++) this.mesh.setColorAt(i, c.setHSL((i * 0.618) % 1, 0.75, 0.62));
  }
  set(data: Float32Array | null): void {
    if (!data) {
      this.mesh.count = 0;
      return;
    }
    const n = Math.min(this.mesh.instanceMatrix.count, Math.floor(data.length / 4));
    for (let i = 0; i < n; i++) {
      tmpV.set(data[i * 4], data[i * 4 + 1], data[i * 4 + 2]);
      tmpQ.setFromAxisAngle(UP, data[i * 4 + 3]);
      tmpM.compose(tmpV, tmpQ, tmpS);
      this.mesh.setMatrixAt(i, tmpM);
    }
    this.count = n;
    this.mesh.count = n;
    this.mesh.instanceMatrix.needsUpdate = true;
  }
}

/** Ray-vision lines + ground feelers of the robot. */
export class SensorLayer {
  group = new THREE.Group();
  private lines: THREE.LineSegments;
  private dots: THREE.InstancedMesh;
  private probes: THREE.InstancedMesh;
  private max = 64;
  constructor() {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.max * 6), 3));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(this.max * 6), 3));
    this.lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9, depthTest: true }));
    this.lines.frustumCulled = false;
    this.dots = new THREE.InstancedMesh(new THREE.SphereGeometry(0.09, 8, 6), new THREE.MeshBasicMaterial({ color: 0xffffff }), this.max);
    this.dots.frustumCulled = false;
    this.probes = new THREE.InstancedMesh(new THREE.SphereGeometry(0.12, 8, 6), new THREE.MeshBasicMaterial({ color: 0xffffff }), 32);
    this.probes.frustumCulled = false;
    this.group.add(this.lines, this.dots, this.probes);
  }
  update(env: Env, ox: number, oy: number, oz: number): void {
    const rv = env.rayVis;
    const n = Math.min(this.max, rv.length / 5);
    const pos = this.lines.geometry.attributes.position as THREE.BufferAttribute;
    const col = this.lines.geometry.attributes.color as THREE.BufferAttribute;
    const c = new THREE.Color();
    for (let i = 0; i < n; i++) {
      const dx = rv[i * 5];
      const dy = rv[i * 5 + 1];
      const dz = rv[i * 5 + 2];
      const t = rv[i * 5 + 3];
      const tag = rv[i * 5 + 4];
      pos.setXYZ(i * 2, ox, oy, oz);
      pos.setXYZ(i * 2 + 1, ox + dx * t, oy + dy * t, oz + dz * t);
      if (tag >= 0) c.set(TAG_INFO[ALL_TAGS[tag]].color);
      else c.setRGB(0.45, 0.5, 0.65);
      col.setXYZ(i * 2, c.r * 0.6, c.g * 0.6, c.b * 0.6);
      col.setXYZ(i * 2 + 1, c.r, c.g, c.b);
      tmpM.makeTranslation(ox + dx * t, oy + dy * t, oz + dz * t);
      this.dots.setMatrixAt(i, tmpM);
      this.dots.setColorAt(i, tag >= 0 ? c : c.setRGB(0.3, 0.32, 0.4));
    }
    this.lines.geometry.setDrawRange(0, n * 2);
    pos.needsUpdate = true;
    col.needsUpdate = true;
    this.dots.count = n;
    this.dots.instanceMatrix.needsUpdate = true;
    if (this.dots.instanceColor) this.dots.instanceColor.needsUpdate = true;
    const pv = env.probeVis;
    const np = Math.min(32, pv.length / 4);
    for (let i = 0; i < np; i++) {
      tmpM.makeTranslation(pv[i * 4], pv[i * 4 + 1] + 0.06, pv[i * 4 + 2]);
      this.probes.setMatrixAt(i, tmpM);
      const k = pv[i * 4 + 3];
      this.probes.setColorAt(i, c.setHex(k === 1 ? 0x222222 : k === 2 ? 0xff4422 : 0x55ff99));
    }
    this.probes.count = np;
    this.probes.instanceMatrix.needsUpdate = true;
    if (this.probes.instanceColor) this.probes.instanceColor.needsUpdate = true;
  }
}

export class Trail {
  line: THREE.Line;
  private n = 0;
  private max = 400;
  private pos: Float32Array;
  constructor() {
    this.pos = new Float32Array(this.max * 3);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    const cols = new Float32Array(this.max * 3);
    for (let i = 0; i < this.max; i++) {
      const k = i / this.max;
      cols.set([0.3 + 0.7 * k, 0.8 * k + 0.2, 1], i * 3);
    }
    g.setAttribute('color', new THREE.BufferAttribute(cols, 3));
    this.line = new THREE.Line(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.75 }));
    this.line.frustumCulled = false;
  }
  clear(): void {
    this.n = 0;
    this.line.geometry.setDrawRange(0, 0);
  }
  push(x: number, y: number, z: number): void {
    if (this.n >= this.max) {
      this.pos.copyWithin(0, 3);
      this.n = this.max - 1;
    }
    this.pos[this.n * 3] = x;
    this.pos[this.n * 3 + 1] = y;
    this.pos[this.n * 3 + 2] = z;
    this.n++;
    const attr = this.line.geometry.attributes.position as THREE.BufferAttribute;
    attr.needsUpdate = true;
    this.line.geometry.setDrawRange(0, this.n);
  }
}
