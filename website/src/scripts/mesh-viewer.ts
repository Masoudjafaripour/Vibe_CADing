// three.js viewer for meshes built in the browser by the C++ engine (WebAssembly).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export function mountMeshViewer(canvas: HTMLCanvasElement) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.5, 5000);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa0ad, 1.9));
  const key = new THREE.DirectionalLight(0xffffff, 2.0);
  key.position.set(0.6, 1, 0.8);
  scene.add(key);
  const grid = new THREE.GridHelper(200, 20, 0x5f6977, 0x5f6977);
  (grid.material as THREE.Material).transparent = true;
  (grid.material as THREE.Material).opacity = 0.3;
  scene.add(grid);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.autoRotate = !matchMedia("(prefers-reduced-motion: reduce)").matches;
  controls.autoRotateSpeed = 1.4;
  controls.addEventListener("start", () => (controls.autoRotate = false));
  camera.position.set(120, 90, 140);

  let mesh: THREE.Mesh | null = null;
  const resize = () => {
    renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    camera.aspect = canvas.clientWidth / canvas.clientHeight;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(canvas);
  resize();
  let visible = true;
  new IntersectionObserver(([e]) => (visible = e.isIntersecting)).observe(canvas);
  renderer.setAnimationLoop(() => {
    if (!visible) return;
    controls.update();
    renderer.render(scene, camera);
  });

  return {
    show(positions: Float32Array, indices: Uint32Array, colors: Float32Array) {
      if (mesh) {
        scene.remove(mesh);
        mesh.geometry.dispose();
      }
      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      geom.setIndex(new THREE.BufferAttribute(indices, 1));
      geom.computeVertexNormals();
      mesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.5, metalness: 0.1 }));
      scene.add(mesh);
      geom.computeBoundingBox();
      const box = geom.boundingBox!;
      const center = box.getCenter(new THREE.Vector3());
      const radius = box.getSize(new THREE.Vector3()).length() / 2;
      grid.position.y = box.min.y;
      grid.scale.setScalar(Math.max(radius / 60, 0.5));
      controls.target.copy(center);
      camera.position.copy(center).add(new THREE.Vector3(1, 0.65, 1.15).normalize().multiplyScalar(radius * 3.1));
    },
    clear() {
      if (mesh) scene.remove(mesh);
      mesh = null;
    },
  };
}
