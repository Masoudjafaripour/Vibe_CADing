// A rubber duck: an organic shape from ellipsoids and spheres joined with
// smooth unions. Faces +Z, rests on y = -1.

float body(vec3 p) {
    float torso = sdEllipsoid(p - vec3(0.0, -0.6, -0.05), vec3(0.48, 0.36, 0.62));
    float tail = sdEllipsoid(rotX(p - vec3(0.0, -0.4, -0.58), 0.6), vec3(0.16, 0.08, 0.16));
    float neck = sdSphere(p - vec3(0.0, -0.25, 0.25), 0.2);
    float head = sdSphere(p - vec3(0.0, 0.05, 0.3), 0.3);
    vec3 q = vec3(abs(p.x), p.y, p.z);  // mirror: one wing models both
    float wing = sdEllipsoid(rotZ(q - vec3(0.42, -0.52, -0.1), -0.25), vec3(0.08, 0.18, 0.32));

    float d = opSmoothUnion(torso, tail, 0.12);
    d = opSmoothUnion(d, wing, 0.06);
    d = opSmoothUnion(d, neck, 0.15);
    return opSmoothUnion(d, head, 0.12);
}

float beak(vec3 p) {
    float upper = sdEllipsoid(p - vec3(0.0, 0.02, 0.6), vec3(0.14, 0.05, 0.13));
    float lower = sdEllipsoid(p - vec3(0.0, -0.03, 0.57), vec3(0.11, 0.035, 0.1));
    return opSmoothUnion(upper, lower, 0.02);
}

float eyes(vec3 p) {
    vec3 q = vec3(abs(p.x), p.y, p.z);
    return sdSphere(q - vec3(0.12, 0.13, 0.55), 0.045);
}

float sdf(vec3 p) {
    return min(min(body(p), beak(p)), eyes(p));
}

vec3 sdf_color(vec3 p) {
    float d = sdf(p) + 1e-3;
    if (eyes(p) <= d) return vec3(0.03);
    if (beak(p) <= d) return vec3(1.0, 0.45, 0.05);
    return vec3(1.0, 0.82, 0.1);
}
