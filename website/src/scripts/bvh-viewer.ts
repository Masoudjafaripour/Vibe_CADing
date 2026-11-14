// Plays the BVH clips exported by the C++ motion engine (vibecad_motion.write_bvh)
// with three.js's own BVHLoader, drawn as a stylised capsule skeleton.
import * as THREE from "three";
import { BVHLoader } from "three/addons/loaders/BVHLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export function mountBvhViewer(canvas: HTMLCanvasElement) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.05, 50);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 2.0));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(2, 4, 3);
  scene.add(key);
  const grid = new THREE.GridHelper(12, 48, 0x5f6977, 0x5f6977);
  (grid.material as THREE.Material).transparent = true;
  (grid.material as THREE.Material).opacity = 0.35;
  scene.add(grid);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.enablePan = false;
  camera.position.set(2.6, 1.6, 3.2);

  const loader = new BVHLoader();
  const timer = new THREE.Timer();
  let mixer: THREE.AnimationMixer | null = null;
  let bones: THREE.Bone[] = [];
  const limbs: { mesh: THREE.Mesh; a: THREE.Bone; b: THREE.Bone }[] = [];
  const joints: { mesh: THREE.Mesh; bone: THREE.Bone }[] = [];
  const rig = new THREE.Group();  // the BVH bones (driven by the mixer)
  const parts = new THREE.Group();  // capsules + joints drawn over them
  scene.add(rig, parts);
  const material = new THREE.MeshStandardMaterial({ color: 0xff6b1f, roughness: 0.5 });
  const up = new THREE.Vector3(0, 1, 0);
  const pa = new THREE.Vector3(), pb = new THREE.Vector3(), mid = new THREE.Vector3(), dir = new THREE.Vector3();

  let visible = true;
  new IntersectionObserver(([e]) => (visible = e.isIntersecting)).observe(canvas);
  const resize = () => {
    renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    camera.aspect = canvas.clientWidth / canvas.clientHeight;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(canvas);
  resize();

  renderer.setAnimationLoop(() => {
    timer.update();
    const dt = timer.getDelta();
    if (!visible || !mixer) return;
    mixer.update(dt);
    rig.updateMatrixWorld(true);
    for (const l of limbs) {
      l.a.getWorldPosition(pa);
      l.b.getWorldPosition(pb);
      mid.addVectors(pa, pb).multiplyScalar(0.5);
      dir.subVectors(pb, pa);
      const len = dir.length();
      l.mesh.position.copy(mid);
      l.mesh.scale.set(1, Math.max(len, 1e-4), 1);
      if (len > 1e-6) l.mesh.quaternion.setFromUnitVectors(up, dir.normalize());
    }
    for (const j of joints) j.bone.getWorldPosition(j.mesh.position);
    // follow the pelvis
    bones[0].getWorldPosition(pa);
    const shift = new THREE.Vector3(pa.x, 0.9, pa.z).sub(controls.target);
    controls.target.add(shift);
    camera.position.add(shift);
    controls.update();
    renderer.render(scene, camera);
  });

  return {
    load(url: string, color: number): Promise<{ frames: number; seconds: number; joints: number }> {
      return new Promise((resolve, reject) => {
        loader.load(url, (result) => {
          rig.clear();
          parts.clear();  // drop the previous clip's body
          mixer?.stopAllAction();
          limbs.length = joints.length = 0;
          bones = result.skeleton.bones;
          rig.add(bones[0]);
          material.color.setHex(color);
          const capsule = new THREE.CylinderGeometry(0.028, 0.028, 1, 10);
          capsule.translate(0, 0, 0);
          const sphere = new THREE.SphereGeometry(0.04, 16, 12);
          for (const b of bones) {
            if (b.name === "ENDSITE") continue;
            const j = new THREE.Mesh(b.name === "head" ? new THREE.SphereGeometry(0.11, 24, 16) : sphere, material);
            parts.add(j);
            joints.push({ mesh: j, bone: b });
            if (b.parent && (b.parent as THREE.Bone).isBone) {
              const m = new THREE.Mesh(capsule, material);
              parts.add(m);
              limbs.push({ mesh: m, a: b.parent as THREE.Bone, b });
            }
          }
          mixer = new THREE.AnimationMixer(bones[0]);
          mixer.clipAction(result.clip).play();
          timer.reset();
          resolve({ frames: result.clip.tracks[0].times.length, seconds: result.clip.duration, joints: joints.length });
        }, undefined, reject);
      });
    },
  };
}
