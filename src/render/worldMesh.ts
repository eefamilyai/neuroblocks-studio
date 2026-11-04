/**
 * Builds the Three.js meshes for the static world: ground, walls, ramps,
 * trees, rocks, lava, water, sky and decorations. Static boxes are merged
 * into a handful of meshes (one draw call each) with per-vertex colours.
 */
import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import type { DecorDef, StaticDef, WorldDef } from '../sim/types.ts';
import { RNG } from '../ml/rng.ts';

const tmpColor = new THREE.Color();

function colorGeometry(g: THREE.BufferGeometry, top: number, side: number, jitter: number, rng: RNG): void {
  const n = g.attributes.normal;
  const count = g.attributes.position.count;
  const cols = new Float32Array(count * 3);
  const k = 1 + (rng.next() - 0.5) * jitter;
  for (let i = 0; i < count; i++) {
    const ny = n.getY(i);
    tmpColor.setHex(ny > 0.5 ? top : side);
    cols[i * 3] = Math.min(1, tmpColor.r * k);
    cols[i * 3 + 1] = Math.min(1, tmpColor.g * k);
    cols[i * 3 + 2] = Math.min(1, tmpColor.b * k);
  }
  g.setAttribute('color', new THREE.BufferAttribute(cols, 3));
}

function darker(hex: number, k: number): number {
  tmpColor.setHex(hex).multiplyScalar(k);
  return tmpColor.getHex();
}

function nonIndexed(g: THREE.BufferGeometry): THREE.BufferGeometry {
  const ng = g.index ? g.toNonIndexed() : g;
  if (ng.attributes.uv) ng.deleteAttribute('uv');
  return ng;
}

export interface WorldMeshes {
  group: THREE.Group;
  gates: THREE.Object3D | null;
  update(t: number): void;
  dispose(): void;
}

export function buildWorldMeshes(world: WorldDef, statics: StaticDef[]): WorldMeshes {
  const group = new THREE.Group();
  const rng = new RNG(99);
  // geometry is merged per zone and material: few draw calls, but zones you
  // are not looking at are skipped (frustum culling), also in the shadow pass
  type Kind = 'matte' | 'shiny' | 'lava' | 'ice' | 'gates';
  const buckets = new Map<string, { kind: Kind; geos: THREE.BufferGeometry[] }>();
  let zoneOf = 'world';
  const bucket = (kind: Kind) => {
    const key = `${zoneOf}|${kind}`;
    let b = buckets.get(key);
    if (!b) buckets.set(key, (b = { kind, geos: [] }));
    return b.geos;
  };
  const matte = { push: (g: THREE.BufferGeometry) => bucket('matte').push(g) };
  const shiny = { push: (g: THREE.BufferGeometry) => bucket('shiny').push(g) };
  const lava = { push: (g: THREE.BufferGeometry) => bucket('lava').push(g) };
  const ice = { push: (g: THREE.BufferGeometry) => bucket('ice').push(g) };
  const gates = { push: (g: THREE.BufferGeometry) => bucket('gates').push(g) };
  const m4 = new THREE.Matrix4();

  for (const s of statics) {
    zoneOf = s.tag === 'ground' && s.kind === 'box' && Math.max(s.h[0], s.h[2]) > 40 ? 'world' : s.zone;
    if (s.kind === 'box') {
      if (s.kind === 'box' && s.invisible) continue;
      const g = new THREE.BoxGeometry(s.h[0] * 2, s.h[1] * 2, s.h[2] * 2);
      const R = s.R;
      m4.set(R[0], R[1], R[2], s.c[0], R[3], R[4], R[5], s.c[1], R[6], R[7], R[8], s.c[2], 0, 0, 0, 1);
      g.applyMatrix4(m4);
      const side = s.tag === 'ground' && (s.mat === 'grass' || s.mat === 'dirt') ? 0x7a5a3c : s.mat === 'hedge' ? darker(s.color, 0.8) : darker(s.color, 0.82);
      const ng = nonIndexed(g);
      colorGeometry(ng, s.color, side, s.tag === 'ground' ? 0.03 : 0.08, rng);
      if (s.gate) gates.push(ng);
      else if (s.mat === 'lava') lava.push(ng);
      else if (s.mat === 'ice') ice.push(ng);
      else if (s.mat === 'metal' || s.mat === 'rubber' || s.mat === 'track') shiny.push(ng);
      else matte.push(ng);
    } else {
      const h = s.y1 - s.y0;
      if (s.tag === 'tree' && s.mat === 'wood') {
        // pine tree: trunk + stacked cones
        const trunk = nonIndexed(new THREE.CylinderGeometry(s.r * 0.75, s.r, h * 0.45, 7).translate(s.c[0], s.y0 + h * 0.225, s.c[1]));
        colorGeometry(trunk, 0x6b4426, 0x6b4426, 0.1, rng);
        matte.push(trunk);
        const green = [0x2e7d32, 0x388e3c, 0x1b5e20][rng.int(3)];
        for (let k = 0; k < 3; k++) {
          const cr = s.r * (4.2 - k * 1.1);
          const ch = h * (0.42 - k * 0.06);
          const cone = nonIndexed(new THREE.ConeGeometry(cr, ch, 8).translate(s.c[0], s.y0 + h * (0.38 + k * 0.2) + ch / 2, s.c[1]));
          colorGeometry(cone, green, darker(green, 0.85), 0.12, rng);
          matte.push(cone);
        }
      } else if (s.tag === 'tree') {
        // bush
        const bush = nonIndexed(new THREE.IcosahedronGeometry(1, 1).scale(s.r * 1.05, h * 0.85, s.r * 1.05).translate(s.c[0], s.y0 + h * 0.5, s.c[1]));
        colorGeometry(bush, s.color, darker(s.color, 0.8), 0.15, rng);
        matte.push(bush);
      } else {
        // boulder / pillar
        const pillar = s.r > 1.1 && h > 2;
        const geo = pillar ? new THREE.CylinderGeometry(s.r, s.r * 1.08, h, 10) : new THREE.DodecahedronGeometry(1, 0).scale(s.r * 1.08, h * 0.62, s.r * 1.08);
        const rock = nonIndexed(geo.rotateY(rng.uniform(0, 6)).translate(s.c[0], pillar ? s.y0 + h / 2 : s.y0 + h * 0.45, s.c[1]));
        colorGeometry(rock, s.color, darker(s.color, 0.78), 0.18, rng);
        matte.push(rock);
      }
    }
  }

  const mats: THREE.Material[] = [];
  const add = (geos: THREE.BufferGeometry[], mat: THREE.Material, shadows = true): THREE.Mesh | null => {
    if (!geos.length) return null;
    const merged = mergeGeometries(geos, false);
    if (!merged) return null;
    merged.computeBoundingSphere();
    const mesh = new THREE.Mesh(merged, mat);
    mesh.castShadow = shadows;
    mesh.receiveShadow = true;
    group.add(mesh);
    geos.forEach((g) => g.dispose());
    return mesh;
  };
  // Lambert/Phong instead of physically-based materials: same flat-shaded look, far cheaper per pixel
  const matteMat = new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true });
  const shinyMat = new THREE.MeshPhongMaterial({ vertexColors: true, flatShading: true, shininess: 40, specular: 0x333333 });
  const iceMat = new THREE.MeshPhongMaterial({ vertexColors: true, shininess: 90, specular: 0x99bbcc, transparent: true, opacity: 0.88 });
  const gateMat = new THREE.MeshLambertMaterial({ vertexColors: true });
  mats.push(matteMat, shinyMat, iceMat, gateMat);
  const gateGroup = new THREE.Group();
  group.add(gateGroup);
  const lavaGeos: THREE.BufferGeometry[] = [];
  for (const b of buckets.values()) {
    if (b.kind === 'lava') lavaGeos.push(...b.geos);
    else if (b.kind === 'gates') {
      const m = add(b.geos, gateMat);
      if (m) gateGroup.add(m);
    } else add(b.geos, b.kind === 'matte' ? matteMat : b.kind === 'shiny' ? shinyMat : iceMat);
  }
  const gateMesh = gateGroup.children.length ? gateGroup : null;

  // lava: animated, self-lit
  const lavaMat = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 } },
    vertexShader: `varying vec3 vP; void main(){ vec4 w = modelMatrix * vec4(position,1.0); vP = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: `uniform float uTime; varying vec3 vP;
      float h(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7)))*43758.5453); }
      float n(vec2 p){ vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
        return mix(mix(h(i),h(i+vec2(1,0)),f.x), mix(h(i+vec2(0,1)),h(i+vec2(1,1)),f.x), f.y); }
      void main(){
        vec2 p = vP.xz*0.35;
        float v = n(p + vec2(uTime*0.15, uTime*0.1)) * 0.6 + n(p*2.3 - vec2(uTime*0.2, -uTime*0.13)) * 0.4;
        vec3 c = mix(vec3(0.55,0.05,0.0), vec3(1.0,0.45,0.05), smoothstep(0.25,0.75,v));
        c = mix(c, vec3(1.0,0.9,0.4), smoothstep(0.78,0.95,v));
        gl_FragColor = vec4(c,1.0);
      }`,
  });
  mats.push(lavaMat);
  add(lavaGeos, lavaMat, false);

  // water around the island
  // the sea covers half the screen: the cheapest lit material, no shadows on it
  const waterMat = new THREE.MeshLambertMaterial({ color: 0x2f8fc7, emissive: 0x0b2a40 });
  const water = new THREE.Mesh(new THREE.PlaneGeometry(2400, 2400, 1, 1).rotateX(-Math.PI / 2), waterMat);
  water.position.y = world.waterY;
  water.receiveShadow = false;
  group.add(water);
  mats.push(waterMat);
  // island skirt (cliffs below the slab down into the water)
  const b = world.bounds;
  const skirt = new THREE.Mesh(
    new THREE.BoxGeometry(b.maxX - b.minX - 0.2, 4.2, b.maxZ - b.minZ - 0.2),
    new THREE.MeshLambertMaterial({ color: 0x6d5038, flatShading: true }),
  );
  skirt.position.set((b.minX + b.maxX) / 2, -6.05, (b.minZ + b.maxZ) / 2);
  group.add(skirt);
  // abyss under the sky zone
  for (const z of world.zones) {
    if (z.id !== 'sky') continue;
    const r = z.rect;
    const abyss = new THREE.Mesh(new THREE.PlaneGeometry(r.maxX - r.minX, r.maxZ - r.minZ).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: 0x0a1630 }));
    abyss.position.set((r.minX + r.maxX) / 2, -3.9, (r.minZ + r.maxZ) / 2);
    group.add(abyss);
    const walls = new THREE.Mesh(
      new THREE.BoxGeometry(r.maxX - r.minX, 3.9, r.maxZ - r.minZ),
      new THREE.MeshLambertMaterial({ color: 0x3d4a66, side: THREE.BackSide }),
    );
    walls.position.set((r.minX + r.maxX) / 2, -1.95, (r.minZ + r.maxZ) / 2);
    group.add(walls);
  }

  addSky(group);
  const decorUpdate = addDecor(group, world, world.decor, rng);
  addZoneMarkings(group, world);

  return {
    group,
    gates: gateMesh,
    update(t: number) {
      lavaMat.uniforms.uTime.value = t;
      water.position.y = world.waterY + Math.sin(t * 0.6) * 0.08;
      decorUpdate(t);
    },
    dispose() {
      const seen = new Set<THREE.Material>(mats);
      group.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.geometry) m.geometry.dispose();
        const mm = m.material as THREE.Material | THREE.Material[] | undefined;
        if (mm) for (const x of Array.isArray(mm) ? mm : [mm]) seen.add(x);
      });
      for (const m of seen) {
        const map = (m as THREE.MeshBasicMaterial).map;
        if (map) map.dispose();
        m.dispose();
      }
    },
  };
}

function addSky(group: THREE.Group): void {
  const sky = new THREE.Mesh(
    new THREE.SphereGeometry(1400, 32, 16),
    new THREE.ShaderMaterial({
      side: THREE.BackSide,
      depthWrite: false,
      fog: false,
      uniforms: {},
      vertexShader: `varying vec3 vW; void main(){ vW = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
      fragmentShader: `varying vec3 vW; void main(){
        float h = vW.y;
        vec3 top = vec3(0.22,0.47,0.86); vec3 hor = vec3(0.80,0.90,1.0); vec3 bot = vec3(0.55,0.70,0.85);
        vec3 c = h > 0.0 ? mix(hor, top, pow(h, 0.55)) : mix(hor, bot, pow(-h, 0.4));
        vec3 sunDir = normalize(vec3(0.45, 0.6, 0.3));
        float s = max(dot(vW, sunDir), 0.0);
        c += vec3(1.0,0.9,0.7) * (pow(s, 600.0) * 2.0 + pow(s, 12.0) * 0.18);
        gl_FragColor = vec4(c, 1.0);
      }`,
    }),
  );
  sky.renderOrder = -1;
  group.add(sky);
}

function addDecor(group: THREE.Group, world: WorldDef, decor: DecorDef[], rng: RNG): (t: number) => void {
  const dummy = new THREE.Object3D();
  const byKind = (k: DecorDef['kind']) => decor.filter((d) => d.kind === k);

  // flowers & tufts (instanced)
  const flowers = byKind('flower');
  if (flowers.length) {
    const heads = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(0.075, 0), new THREE.MeshLambertMaterial({ emissive: 0x222222 }), flowers.length);
    const stems = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.012, 0.012, 0.3, 4), new THREE.MeshLambertMaterial({ color: 0x3f8f2f }), flowers.length);
    flowers.forEach((f, i) => {
      const s = f.s ?? 1;
      dummy.position.set(f.x, f.y + 0.35 * s, f.z);
      dummy.scale.setScalar(s);
      dummy.rotation.set(0, 0, 0);
      dummy.updateMatrix();
      heads.setMatrixAt(i, dummy.matrix);
      heads.setColorAt(i, tmpColor.setHex(f.color ?? 0xffffff));
      dummy.position.set(f.x, f.y + 0.17 * s, f.z);
      dummy.updateMatrix();
      stems.setMatrixAt(i, dummy.matrix);
    });
    group.add(heads, stems);
  }
  const tufts = byKind('tuft');
  if (tufts.length) {
    const m = new THREE.InstancedMesh(new THREE.ConeGeometry(0.16, 0.45, 4), new THREE.MeshLambertMaterial({ flatShading: true }), tufts.length * 3);
    let k = 0;
    tufts.forEach((t) => {
      for (let j = 0; j < 3; j++) {
        dummy.position.set(t.x + rng.uniform(-0.2, 0.2), t.y + 0.2, t.z + rng.uniform(-0.2, 0.2));
        dummy.rotation.set(rng.uniform(-0.3, 0.3), 0, rng.uniform(-0.3, 0.3));
        dummy.scale.setScalar(t.s ?? 1);
        dummy.updateMatrix();
        m.setMatrixAt(k, dummy.matrix);
        m.setColorAt(k, tmpColor.setHex(t.color ?? 0x5aa83a).multiplyScalar(rng.uniform(0.85, 1.1)));
        k++;
      }
    });
    group.add(m);
  }
  const bushes = byKind('bush');
  if (bushes.length) {
    const m = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 1), new THREE.MeshLambertMaterial({ flatShading: true }), bushes.length);
    bushes.forEach((bb, i) => {
      const s = bb.s ?? 1;
      dummy.position.set(bb.x, bb.y + s * 0.55, bb.z);
      dummy.scale.set(s, s * 0.8, s);
      dummy.rotation.set(0, rng.uniform(0, 6), 0);
      dummy.updateMatrix();
      m.setMatrixAt(i, dummy.matrix);
      m.setColorAt(i, tmpColor.setHex(bb.color ?? 0x3d8b3d));
    });
    m.castShadow = true;
    group.add(m);
  }
  // lamps
  for (const l of byKind('lamp')) {
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.12, 3.2, 6), new THREE.MeshLambertMaterial({ color: 0x3a3f55 }));
    post.position.set(l.x, 1.6, l.z);
    const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.25, 12, 8), new THREE.MeshLambertMaterial({ color: 0xfff1c2, emissive: 0xffd27a, emissiveIntensity: 1.2 }));
    bulb.position.set(l.x, 3.3, l.z);
    post.castShadow = true;
    group.add(post, bulb);
  }
  // paths
  for (const p of byKind('path')) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(p.w ?? 4, p.d ?? 4).rotateX(-Math.PI / 2), new THREE.MeshLambertMaterial({ color: p.color ?? 0xcfb58a }));
    m.position.set(p.x, p.y, p.z);
    m.receiveShadow = true;
    group.add(m);
  }
  // mountains
  const mgeos: THREE.BufferGeometry[] = [];
  for (const mt of byKind('mountain')) {
    const s = mt.s ?? 40;
    const g = nonIndexed(new THREE.ConeGeometry(s, s * 1.25, 7, 3).translate(mt.x, mt.y + s * 0.62, mt.z));
    const pos = g.attributes.position;
    const cols = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
      const y = (pos.getY(i) - mt.y) / (s * 1.25);
      tmpColor.setHex(y > 0.72 ? 0xf4f7fb : mt.color ?? 0x7a8fa6);
      cols.set([tmpColor.r, tmpColor.g, tmpColor.b], i * 3);
    }
    g.setAttribute('color', new THREE.BufferAttribute(cols, 3));
    mgeos.push(g);
  }
  if (mgeos.length) {
    const merged = mergeGeometries(mgeos, false);
    if (merged) group.add(new THREE.Mesh(merged, new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true })));
  }
  // clouds (drift slowly)
  const clouds = new THREE.Group();
  const cgeos: THREE.BufferGeometry[] = [];
  for (const c of byKind('cloud')) {
    const s = c.s ?? 10;
    for (let k = 0; k < 5; k++) {
      cgeos.push(
        nonIndexed(
          new THREE.IcosahedronGeometry(s * rng.uniform(0.35, 0.6), 1)
            .scale(1, 0.6, 1)
            .translate(c.x + rng.uniform(-s, s), c.y + rng.uniform(-s * 0.15, s * 0.15), c.z + rng.uniform(-s * 0.5, s * 0.5)),
        ),
      );
    }
  }
  if (cgeos.length) {
    const merged = mergeGeometries(cgeos, false);
    if (merged) clouds.add(new THREE.Mesh(merged, new THREE.MeshLambertMaterial({ color: 0xffffff, flatShading: true, transparent: true, opacity: 0.92, emissive: 0x9aa8c8, emissiveIntensity: 0.25 })));
  }
  group.add(clouds);
  // tyre stacks
  for (const bb of byKind('barrel')) {
    for (let k = 0; k < 3; k++) {
      const tyre = new THREE.Mesh(new THREE.TorusGeometry(0.45, 0.2, 6, 12).rotateX(Math.PI / 2), new THREE.MeshLambertMaterial({ color: k % 2 ? 0xd33 : 0x222 }));
      tyre.position.set(bb.x, 0.2 + k * 0.38, bb.z);
      tyre.castShadow = true;
      group.add(tyre);
    }
  }
  // start banner
  for (const bn of byKind('banner')) {
    const w = bn.w ?? 10;
    const postMat = new THREE.MeshLambertMaterial({ color: 0xdddddd });
    for (const sx of [-1, 1]) {
      const post = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 4.2, 8), postMat);
      post.position.set(bn.x, 2.1, bn.z + (sx * w) / 2);
      group.add(post);
    }
    const tex = textTexture(bn.text ?? 'START', 512, 96, '#ffffff', '#c62828');
    for (const side of [1, -1]) {
      const board = new THREE.Mesh(new THREE.PlaneGeometry(w, 1.1), new THREE.MeshBasicMaterial({ map: tex }));
      board.rotation.y = (side * Math.PI) / 2;
      board.position.set(bn.x + side * 0.02, 4.0, bn.z);
      group.add(board);
    }
    // checkered line
    const check = checkerTexture();
    const line = new THREE.Mesh(new THREE.PlaneGeometry(1.2, w).rotateX(-Math.PI / 2), new THREE.MeshLambertMaterial({ map: check }));
    line.position.set(bn.x, 0.03, bn.z);
    group.add(line);
  }
  // soccer net
  for (const nt of byKind('net')) {
    const w = nt.w ?? 9;
    const d = nt.d ?? 3.5;
    const h = nt.h ?? 2.6;
    const box = new THREE.BoxGeometry(w, h, d, 18, 6, 7);
    const wire = new THREE.LineSegments(new THREE.WireframeGeometry(box), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35 }));
    wire.position.set(nt.x, h / 2, nt.z);
    group.add(wire);
    const barMat = new THREE.MeshLambertMaterial({ color: 0xffffff });
    const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, w + 0.2, 8).rotateZ(Math.PI / 2), barMat);
    bar.position.set(nt.x, h, nt.z + d / 2);
    group.add(bar);
    for (const sx of [-1, 1]) {
      const post = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, h, 8), barMat);
      post.position.set(nt.x + (sx * w) / 2, h / 2, nt.z + d / 2);
      group.add(post);
    }
  }
  return (t: number) => {
    clouds.position.x = Math.sin(t * 0.01) * 40;
    clouds.position.z = Math.cos(t * 0.013) * 25;
  };
}

function addZoneMarkings(group: THREE.Group, world: WorldDef): void {
  const white = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.85 });
  const soccer = world.zones.find((z) => z.id === 'soccer');
  if (soccer) {
    const cx = (soccer.rect.minX + soccer.rect.maxX) / 2;
    const cz = (soccer.rect.minZ + soccer.rect.maxZ) / 2;
    const strip = (x: number, z: number, w: number, d: number) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, d).rotateX(-Math.PI / 2), white);
      m.position.set(cx + x, 0.025, cz + z);
      group.add(m);
    };
    strip(0, 0, 32, 0.15);
    strip(-15.85, 0, 0.15, 36);
    strip(15.85, 0, 0.15, 36);
    strip(0, -12, 14, 0.15);
    strip(-7, -15, 0.15, 6);
    strip(7, -15, 0.15, 6);
    const ring = new THREE.Mesh(new THREE.RingGeometry(3.9, 4.05, 48).rotateX(-Math.PI / 2), white);
    ring.position.set(cx, 0.025, cz);
    group.add(ring);
  }
}

export function textTexture(text: string, w: number, h: number, fg: string, bg: string): THREE.CanvasTexture {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const g = c.getContext('2d')!;
  g.fillStyle = bg;
  g.fillRect(0, 0, w, h);
  g.fillStyle = fg;
  g.font = `800 ${Math.floor(h * 0.62)}px Inter, system-ui, sans-serif`;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(text, w / 2, h / 2 + 2);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function checkerTexture(): THREE.CanvasTexture {
  const c = document.createElement('canvas');
  c.width = 64;
  c.height = 512;
  const g = c.getContext('2d')!;
  for (let y = 0; y < 32; y++)
    for (let x = 0; x < 4; x++) {
      g.fillStyle = (x + y) % 2 ? '#111' : '#f5f5f5';
      g.fillRect(x * 16, y * 16, 16, 16);
    }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
