// three.js viewer for a B-rep part tessellated from the repo's STEP file
// by website/scripts/step_to_mesh.py (faces as triangles, B-rep edges as lines).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

interface MeshData {
  bbox: number[];
  faces: number;
  edges_count: number;
  positions: number[];
  indices: number[];
  edges: number[][];
}

export async function mountBrepViewer(canvas: HTMLCanvasElement, url: string) {
  const data: MeshData = await (await fetch(url)).json();

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 1000);
  camera.up.set(0, 0, 1); // CAD convention: Z up

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.Float32BufferAttribute(data.positions, 3));
  geom.setIndex(data.indices);
  geom.computeVertexNormals();
  const mesh = new THREE.Mesh(
    geom,
    new THREE.MeshStandardMaterial({ color: 0xc89b1e, metalness: 0.15, roughness: 0.55, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }),
  );
  scene.add(mesh);

  const segs: number[] = [];
  for (const pl of data.edges) {
    for (let i = 0; i + 5 < pl.length; i += 3) segs.push(...pl.slice(i, i + 6));
  }
  const edgeGeom = new THREE.BufferGeometry();
  edgeGeom.setAttribute("position", new THREE.Float32BufferAttribute(segs, 3));
  const edgeMat = new THREE.LineBasicMaterial({ color: 0x15171c });
  scene.add(new THREE.LineSegments(edgeGeom, edgeMat));

  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 1.6));
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(1, -1.5, 2);
  scene.add(key);

  const [x0, y0, z0, x1, y1, z1] = data.bbox;
  const center = new THREE.Vector3((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
  const radius = Math.hypot(x1 - x0, y1 - y0, z1 - z0) / 2;
  camera.position.copy(center).add(new THREE.Vector3(1.1, -1.6, 1.2).normalize().multiplyScalar(radius * 3.4));

  const controls = new OrbitControls(camera, canvas);
  controls.target.copy(center);
  controls.enableDamping = true;
  controls.enableZoom = true;
  controls.autoRotate = !matchMedia("(prefers-reduced-motion: reduce)").matches;
  controls.autoRotateSpeed = 1.2;
  controls.addEventListener("start", () => (controls.autoRotate = false));

  const applyTheme = () => {
    const dark = matchMedia("(prefers-color-scheme: dark)").matches;
    edgeMat.color.set(dark ? 0xe9eaee : 0x15171c);
  };
  applyTheme();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);

  let visible = true;
  new IntersectionObserver(([e]) => (visible = e.isIntersecting)).observe(canvas);

  const resize = () => {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
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
    stats: `${(x1 - x0).toFixed(0)} × ${(y1 - y0).toFixed(0)} × ${(z1 - z0).toFixed(0)} mm · ${data.faces} B-rep faces · ${data.edges_count} edges · ${data.indices.length / 3} triangles`,
    reset: () => {
      camera.position.copy(center).add(new THREE.Vector3(1.1, -1.6, 1.2).normalize().multiplyScalar(radius * 3.4));
      controls.target.copy(center);
    },
  };
}
