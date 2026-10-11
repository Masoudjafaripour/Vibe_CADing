// A snowman: three smoothly blended snow spheres, coal eyes and buttons,
// a carrot nose, stick arms and a top hat. Faces +Z, rests on y = -1.

float snow(vec3 p) {
    float base  = sdSphere(p - vec3(0.0, -0.55, 0.0), 0.45);
    float torso = sdSphere(p - vec3(0.0,  0.15, 0.0), 0.33);
    float head  = sdSphere(p - vec3(0.0,  0.62, 0.0), 0.23);
    return opSmoothUnion(opSmoothUnion(base, torso, 0.08), head, 0.06);
}

float coal(vec3 p) {
    vec3 q = vec3(abs(p.x), p.y, p.z);  // mirror: one eye models both
    float eyes = sdSphere(q - vec3(0.08, 0.68, 0.195), 0.03);
    float buttons = min(sdSphere(p - vec3(0.0, 0.25, 0.305), 0.035),
                    min(sdSphere(p - vec3(0.0, 0.12, 0.318), 0.035),
                        sdSphere(p - vec3(0.0, -0.01, 0.28), 0.035)));
    return min(eyes, buttons);
}

float carrot(vec3 p) {
    // the cone's axis is +Y; rotate it to point along +Z
    return sdRoundCone(rotX(p - vec3(0.0, 0.62, 0.2), -1.5708), 0.04, 0.008, 0.2);
}

float sticks(vec3 p) {
    vec3 q = vec3(abs(p.x), p.y, p.z);
    float arm = sdCapsule(q, vec3(0.28, 0.2, 0.0), vec3(0.62, 0.42, 0.0), 0.02);
    float twig = sdCapsule(q, vec3(0.52, 0.36, 0.0), vec3(0.6, 0.5, 0.02), 0.012);
    return min(arm, twig);
}

float hat(vec3 p) {
    float crown = sdCylinder(p - vec3(0.0, 0.92, 0.0), 0.14, 0.09);
    float brim = sdCylinder(p - vec3(0.0, 0.83, 0.0), 0.24, 0.012);
    return min(crown, brim);
}

float sdf(vec3 p) {
    return min(min(snow(p), coal(p)), min(min(carrot(p), sticks(p)), hat(p)));
}

vec3 sdf_color(vec3 p) {
    float d = sdf(p) + 1e-3;
    if (coal(p) <= d) return vec3(0.04);
    if (carrot(p) <= d) return vec3(0.95, 0.4, 0.08);
    if (sticks(p) <= d) return vec3(0.35, 0.2, 0.1);
    if (hat(p) <= d) return vec3(0.08, 0.08, 0.1);
    return vec3(0.95, 0.97, 1.0);
}
