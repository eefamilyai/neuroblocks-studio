/**
 * The 3D viewport: renderer, camera modes, lights and picking.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export type CamMode = 'follow' | 'orbit' | 'zone' | 'top' | 'free';

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
  lowPower = false;

  constructor(el: HTMLElement) {
    this.el = el;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
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

    const hemi = new THREE.HemisphereLight(0xcfe8ff, 0x5d7a3a, 1.1);
    this.scene.add(hemi);
    this.sun = new THREE.DirectionalLight(0xfff0d8, 2.4);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    const sc = this.sun.shadow.camera;
    sc.left = -45;
    sc.right = 45;
    sc.top = 45;
    sc.bottom = -45;
    sc.near = 1;
    sc.far = 260;
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.03;
    this.scene.add(this.sun, this.sun.target);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.15));

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.minDistance = 2;
    this.controls.maxDistance = 420;
    this.controls.target.set(-28, 0, 14);
    this.controls.addEventListener('start', () => {
      this.fly = null;
    });

    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(el);
    this.resize();
    this.renderer.setAnimationLoop(() => this.frame());
  }

  resize(): void {
    const w = Math.max(1, this.el.clientWidth);
    const h = Math.max(1, this.el.clientHeight);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
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
    const dt = Math.min(0.1, (nowT - this.lastT) / 1000);
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
      const delta = this.focus.clone().sub(this.controls.target).multiplyScalar(Math.min(1, dt * 6));
      this.controls.target.add(delta);
      this.camera.position.add(delta);
      if (this.mode === 'follow') {
        // swing gently behind the robot
        const off = this.camera.position.clone().sub(this.controls.target);
        const cur = Math.atan2(off.x, off.z);
        const want = this.focusYaw + Math.PI;
        let d = want - cur;
        while (d > Math.PI) d -= Math.PI * 2;
        while (d < -Math.PI) d += Math.PI * 2;
        const rot = d * Math.min(1, dt * 1.6);
        off.applyAxisAngle(new THREE.Vector3(0, 1, 0), rot);
        this.camera.position.copy(this.controls.target).add(off);
      }
    }
    this.controls.update();
    // shadows follow what we look at
    const tg = this.controls.target;
    this.sun.position.set(tg.x + 60, tg.y + 95, tg.z + 40);
    this.sun.target.position.copy(tg);
    this.renderer.render(this.scene, this.camera);
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
    const v = new THREE.Vector3(x, y, z).project(this.camera);
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
