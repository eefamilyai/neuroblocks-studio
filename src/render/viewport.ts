/**
 * The 3D viewport: renderer, camera modes, lights and picking.
 *
 * Performance: the render resolution adapts to the measured frame time
 * ("auto" quality), shadows can be smaller or off, the frame rate can be
 * capped, and when nothing moves the view redraws only a few times a second.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export type CamMode = 'follow' | 'orbit' | 'zone' | 'top' | 'free';
export type Quality = 'auto' | 'high' | 'medium' | 'low';

const SHADOW: Record<Exclude<Quality, 'auto'>, number> = { high: 2048, medium: 1024, low: 0 };

export class Viewport {
  readonly el: HTMLElement;
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  readonly sun: THREE.DirectionalLight;
  mode: CamMode = 'follow';
  /** where the camera wants to look (robot / zone centre) */
  focus = new THREE.Vector3();
  focusYaw = 0;
  private fly: { from: THREE.Vector3; to: THREE.Vector3; camFrom: THREE.Vector3; camTo: THREE.Vector3; t: number } | null = null;
  private ray = new THREE.Raycaster();
  onFrame: ((dt: number, t: number) => void) | null = null;
  private lastT = performance.now();
  private elapsed = 0;
  pickables: THREE.Object3D[] = [];
  private ro: ResizeObserver;
  // performance
  quality: Quality = 'auto';
  /** max frames per second (0 = every display refresh) */
  maxFps = 0;
  /** set by the app when nothing in the scene is moving */
  idle = false;
  private dprMax = Math.min(window.devicePixelRatio || 1, 2);
  private dpr = 1;
  private frameMs = 16;
  private slowFor = 0;
  private fastFor = 0;
  private lastRender = 0;
  private interacting = false;
  fps = 60;
  /** redraw the shadow map every N frames (static world, so every other frame is plenty) */
  private shadowEvery = 1;
  private frameNo = 0;
  /** auto quality turned shadows off because even the lowest resolution was too slow */
  private autoNoShadows = false;
  private fpsFrames = 0;
  private fpsT0 = performance.now();
  /** camera moved since the last time the app asked (for overlay labels) */
  cameraMoved = true;
  private lastCam = new THREE.Matrix4();
  private tmpV = new THREE.Vector3();
  private tmpOff = new THREE.Vector3();
  private readonly up = new THREE.Vector3(0, 1, 0);

  constructor(el: HTMLElement, quality: Quality = 'auto') {
    this.el = el;
    this.quality = quality;
    const hiDpi = (window.devicePixelRatio || 1) >= 1.5;
    // MSAA is costly and barely visible on high-DPI screens; FXAA-free crisp edges come from the DPR there
    this.renderer = new THREE.WebGLRenderer({ antialias: quality === 'high' || (quality !== 'low' && !hiDpi), powerPreference: 'high-performance', preserveDrawingBuffer: false });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    el.appendChild(this.renderer.domElement);
    this.renderer.domElement.tabIndex = 0;

    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 3000);
    this.camera.position.set(-28, 14, 28);
    this.scene.fog = new THREE.Fog(0xcfe4ff, 160, 900);
    this.scene.background = new THREE.Color(0xcfe4ff);

    const hemi = new THREE.HemisphereLight(0xcfe8ff, 0x5d7a3a, 1.25);
    this.scene.add(hemi);
    this.sun = new THREE.DirectionalLight(0xfff0d8, 2.4);
    this.sun.castShadow = true;
    const sc = this.sun.shadow.camera;
    sc.left = -40;
    sc.right = 40;
    sc.top = 40;
    sc.bottom = -40;
    sc.near = 1;
    sc.far = 260;
    this.sun.shadow.bias = -0.0005;
    this.sun.shadow.normalBias = 0.04;
    this.scene.add(this.sun, this.sun.target);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.18));

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.minDistance = 2;
    this.controls.maxDistance = 420;
    this.controls.target.set(-28, 0, 14);
    this.controls.addEventListener('start', () => {
      this.fly = null;
      this.interacting = true;
    });
    this.controls.addEventListener('end', () => (this.interacting = false));
    this.controls.addEventListener('change', () => (this.cameraMoved = true));

    this.setQuality(quality);
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(el);
    this.resize();
    this.renderer.setAnimationLoop(() => this.frame());
  }

  setQuality(q: Quality): void {
    this.quality = q;
    this.autoNoShadows = false;
    const shadow = q === 'auto' ? 1024 : SHADOW[q];
    this.shadowEvery = q === 'high' ? 1 : 2;
    this.setShadows(shadow);
    this.dpr = q === 'high' ? this.dprMax : q === 'medium' ? Math.min(this.dprMax, 1.25) : q === 'low' ? Math.min(1, this.dprMax) * 0.75 : Math.min(this.dprMax, 1.5);
    this.applyDpr();
  }

  private setShadows(shadow: number): void {
    this.renderer.shadowMap.enabled = shadow > 0;
    this.sun.castShadow = shadow > 0;
    if (shadow > 0 && this.sun.shadow.mapSize.x !== shadow) {
      this.sun.shadow.mapSize.set(shadow, shadow);
      this.sun.shadow.map?.dispose();
      this.sun.shadow.map = null as unknown as THREE.WebGLRenderTarget;
    }
    // materials compile differently with/without shadows
    this.scene.traverse((o) => {
      const m = (o as THREE.Mesh).material as THREE.Material | THREE.Material[] | undefined;
      if (m) for (const mm of Array.isArray(m) ? m : [m]) mm.needsUpdate = true;
    });
  }

  private applyDpr(): void {
    this.renderer.setPixelRatio(this.dpr);
    this.resize();
  }

  /** shadows currently on? */
  get shadows(): boolean {
    return this.renderer.shadowMap.enabled;
  }

  /** current render scale (for the settings dialog) */
  get pixelRatio(): number {
    return this.dpr;
  }

  resize(): void {
    const w = Math.max(1, this.el.clientWidth);
    const h = Math.max(1, this.el.clientHeight);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.cameraMoved = true;
  }

  setMode(m: CamMode): void {
    this.mode = m;
    this.fly = null;
    if (m === 'top') this.flyTo(this.focus, 60, Math.PI * 0.001);
    if (m === 'follow') this.flyTo(this.focus, 11, 1.12);
  }

  /** Smooth camera flight to look at `p` from `dist` metres at `polar` (rad from vertical). */
  flyTo(p: THREE.Vector3, dist = 40, polar = 0.85, azimuth?: number): void {
    const off = this.camera.position.clone().sub(this.controls.target);
    const az = azimuth ?? Math.atan2(off.x, off.z);
    const camTo = new THREE.Vector3(p.x + Math.sin(polar) * Math.sin(az) * dist, p.y + Math.cos(polar) * dist, p.z + Math.sin(polar) * Math.cos(az) * dist);
    this.fly = { from: this.controls.target.clone(), to: p.clone(), camFrom: this.camera.position.clone(), camTo, t: 0 };
  }

  private frame(): void {
    const nowT = performance.now();
    const rawDt = (nowT - this.lastT) / 1000;
    // frame cap / idle throttling (keeps laptops cool when nothing moves)
    const cap = this.idle && !this.interacting && !this.fly ? 8 : this.maxFps;
    if (cap > 0 && nowT - this.lastRender < 1000 / cap - 2) return;
    if (!this.el.offsetParent) return; // hidden (mobile tab)
    const dt = Math.min(0.1, rawDt);
    this.lastT = nowT;
    this.elapsed += dt;
    if (this.onFrame) this.onFrame(dt, this.elapsed);
    if (this.fly) {
      const f = this.fly;
      f.t = Math.min(1, f.t + dt * 1.4);
      const e = f.t < 0.5 ? 4 * f.t * f.t * f.t : 1 - Math.pow(-2 * f.t + 2, 3) / 2;
      this.controls.target.lerpVectors(f.from, f.to, e);
      this.camera.position.lerpVectors(f.camFrom, f.camTo, e);
      if (f.t >= 1) this.fly = null;
    } else if (this.mode === 'follow' || this.mode === 'orbit') {
      // keep the camera offset, move with the robot
      const delta = this.tmpV.copy(this.focus).sub(this.controls.target).multiplyScalar(Math.min(1, dt * 6));
      this.controls.target.add(delta);
      this.camera.position.add(delta);
      if (this.mode === 'follow' && !this.interacting) {
        // swing gently behind the robot
        const off = this.tmpOff.copy(this.camera.position).sub(this.controls.target);
        const cur = Math.atan2(off.x, off.z);
        let d = this.focusYaw + Math.PI - cur;
        while (d > Math.PI) d -= Math.PI * 2;
        while (d < -Math.PI) d += Math.PI * 2;
        off.applyAxisAngle(this.up, d * Math.min(1, dt * 1.6));
        this.camera.position.copy(this.controls.target).add(off);
      }
    }
    this.controls.update();
    // shadows follow what we look at; a tighter box when close gives sharper shadows
    const tg = this.controls.target;
    const dist = this.camera.position.distanceTo(tg);
    const half = Math.max(22, Math.min(70, dist * 1.8));
    const sc = this.sun.shadow.camera;
    if (Math.abs(sc.right - half) > 2) {
      sc.left = -half;
      sc.right = half;
      sc.top = half;
      sc.bottom = -half;
      sc.updateProjectionMatrix();
    }
    this.sun.position.set(tg.x + 60, tg.y + 95, tg.z + 40);
    this.sun.target.position.copy(tg);
    if (!this.lastCam.equals(this.camera.matrixWorld)) {
      this.cameraMoved = true;
      this.lastCam.copy(this.camera.matrixWorld);
    }
    this.frameNo++;
    this.renderer.shadowMap.autoUpdate = false;
    this.renderer.shadowMap.needsUpdate = this.shadowEvery <= 1 || this.frameNo % this.shadowEvery === 0 || !!this.fly || this.interacting;
    this.renderer.render(this.scene, this.camera);
    this.lastRender = nowT;
    this.adapt(rawDt * 1000);
  }

  /** Auto quality: lower the resolution when frames are slow, raise it again when there is headroom. */
  private adapt(ms: number): void {
    this.fpsFrames++;
    const now = performance.now();
    if (now - this.fpsT0 > 1000) {
      this.fps = (this.fpsFrames * 1000) / (now - this.fpsT0);
      this.fpsFrames = 0;
      this.fpsT0 = now;
    }
    if (this.quality !== 'auto' || this.maxFps || this.idle || ms > 250) return;
    this.frameMs = this.frameMs * 0.9 + ms * 0.1;
    if (this.frameMs > 24) {
      this.slowFor += ms;
      this.fastFor = 0;
    } else if (this.frameMs < 14) {
      this.fastFor += ms;
      this.slowFor = 0;
    } else {
      this.slowFor = this.fastFor = 0;
    }
    if (this.slowFor > 1200 && this.dpr > 0.6) {
      this.dpr = Math.max(0.6, +(this.dpr - 0.15).toFixed(2));
      this.slowFor = 0;
      this.applyDpr();
    } else if (this.slowFor > 2500 && !this.autoNoShadows) {
      // the lowest resolution is still too slow: shadows are the next most expensive thing
      this.autoNoShadows = true;
      this.slowFor = 0;
      this.setShadows(0);
    } else if (this.fastFor > 4000 && this.autoNoShadows && this.dpr >= 0.9) {
      this.autoNoShadows = false;
      this.fastFor = 0;
      this.setShadows(1024);
    } else if (this.fastFor > 4000 && this.dpr < Math.min(this.dprMax, 1.5)) {
      this.dpr = Math.min(this.dprMax, 1.5, +(this.dpr + 0.1).toFixed(2));
      this.fastFor = 0;
      this.applyDpr();
    }
  }

  /** World point under the mouse (on pickable meshes), or null. */
  pick(clientX: number, clientY: number): THREE.Vector3 | null {
    const r = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    this.ray.setFromCamera(ndc, this.camera);
    const hits = this.ray.intersectObjects(this.pickables, false);
    if (hits.length) return hits[0].point;
    // fall back to the ground plane y = 0
    const p = new THREE.Vector3();
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    return this.ray.ray.intersectPlane(plane, p);
  }

  /** Project a world point to pixel coordinates inside the viewport (or null if behind). */
  project(x: number, y: number, z: number): [number, number] | null {
    const v = this.tmpV.set(x, y, z).project(this.camera);
    if (v.z > 1 || v.z < -1) return null;
    return [((v.x + 1) / 2) * this.el.clientWidth, ((1 - v.y) / 2) * this.el.clientHeight];
  }

  screenshot(): string {
    this.renderer.render(this.scene, this.camera);
    return this.renderer.domElement.toDataURL('image/png');
  }

  dispose(): void {
    this.ro.disconnect();
    this.renderer.setAnimationLoop(null);
    this.renderer.dispose();
  }
}
