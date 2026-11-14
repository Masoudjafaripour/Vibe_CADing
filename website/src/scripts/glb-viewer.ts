// three.js viewer for the gallery's coloured .glb models (exported by the C++ engine).
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export function mountGlbViewer(canvas: HTMLCanvasElement) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.001, 100);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa0ad, 1.8));
  const key = new THREE.DirectionalLight(0xffffff, 2.0);
  key.position.set(0.6, 1, 0.8);
  scene.add(key);
  const ground = new THREE.Mesh(
    new THREE.CircleGeometry(1, 64),
    new THREE.MeshBasicMaterial({ color: 0x8a8f99, transparent: true, opacity: 0.12 }),
  );
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.autoRotate = !matchMedia("(prefers-reduced-motion: reduce)").matches;
  controls.autoRotateSpeed = 1.5;
  controls.addEventListener("start", () => (controls.autoRotate = false));

  const loader = new GLTFLoader();
  let current: THREE.Object3D | null = null;
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
    if (!visible) return;
    controls.update();
    renderer.render(scene, camera);
  });

  return {
    load(url: string): Promise<{ triangles: number }> {
      return new Promise((resolve, reject) => {
        loader.load(
          url,
          (gltf) => {
            if (current) scene.remove(current);
            current = gltf.scene;
            let triangles = 0;
            current.traverse((o) => {
              const mesh = o as THREE.Mesh;
              if (mesh.isMesh) {
                const mat = mesh.material as THREE.MeshStandardMaterial;
                mat.side = THREE.FrontSide;
                triangles += (mesh.geometry.index?.count ?? 0) / 3;
              }
            });
            scene.add(current);
            const box = new THREE.Box3().setFromObject(current);
            const size = box.getSize(new THREE.Vector3());
            const center = box.getCenter(new THREE.Vector3());
            const radius = size.length() / 2;
            ground.scale.setScalar(radius * 1.05);
            ground.position.set(center.x, box.min.y - 1e-4, center.z);
            controls.target.copy(center);
            camera.position.copy(center).add(new THREE.Vector3(1, 0.55, 1.25).normalize().multiplyScalar(radius * 3.3));
            camera.near = radius / 50;
            camera.far = radius * 50;
            camera.updateProjectionMatrix();
            resolve({ triangles });
          },
          undefined,
          reject,
        );
      });
    },
  };
}
