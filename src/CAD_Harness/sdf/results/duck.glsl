float sphere(vec3 p, float r) {
    return length(p) - r;
}

float box(vec3 p, vec3 b) {
    vec3 q = abs(p);
    return max(q.x - b.x, max(q.y - b.y, q.z - b.z));
}

float cylinder(vec3 p, float r, float h) {
    vec3 q = abs(p);
    float d = max(q.x - r, q.z - r);
    return max(d - h * 0.5, q.y - h * 0.5);
}

float torus(vec3 p, float R, float r) {
    vec2 q = vec2(length(p.xz) - R, p.y);
    return length(q) - r;
}

float sdf(vec3 p) {
    // Base shape: a sphere for the body
    float body = sphere(p, 0.5);
    
    // Head: a smaller sphere on top
    float head = sphere(p - vec3(0, 0.3, 0), 0.2);
    
    // Beak: a cylinder pointing down
    float beak = cylinder(p - vec3(0, 0.3, 0), 0.05, 0.2);
    
    // Eyes: two spheres
    float eye1 = sphere(p - vec3(0.15, 0.4, 0), 0.05);
    float eye2 = sphere(p - vec3(-0.15, 0.4, 0), 0.05);
    
    // Nose: a small sphere
    float nose = sphere(p - vec3(0, 0.35, 0), 0.05);
    
    // Combine body and head with union
    float body_head = min(body, head);
    
    // Combine body_head with beak using subtraction
    float duck = max(body_head, -beak);
    
    // Combine with eyes using union
    duck = min(duck, eye1);
    duck = min(duck, eye2);
    
    // Combine with nose using union
    duck = min(duck, nose);
    
    return duck;
}