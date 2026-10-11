float sdDragonBody(vec3 p) {
    // Main body as a series of capsules
    float body = sdCapsule(p, vec3(-0.6, -0.2, 0.0), vec3(0.6, -0.2, 0.0), 0.15);
    float neck = sdCapsule(p, vec3(-0.6, -0.2, 0.0), vec3(-0.8, 0.3, 0.0), 0.1);
    float tail = sdCapsule(p, vec3(0.6, -0.2, 0.0), vec3(0.9, -0.4, 0.0), 0.08);
    return opSmoothUnion(opSmoothUnion(body, neck, 0.1), tail, 0.1);
}

float sdDragonHead(vec3 p) {
    // Head as a rounded box
    vec3 headPos = vec3(-0.85, 0.35, 0.0);
    float head = sdRoundBox(p - headPos, vec3(0.15, 0.1, 0.1), 0.05);
    // Eyes as small spheres
    float eye1 = sdSphere(p - (headPos + vec3(0.05, 0.05, 0.1)), 0.03);
    float eye2 = sdSphere(p - (headPos + vec3(0.05, 0.05, -0.1)), 0.03);
    return opSmoothUnion(opSmoothUnion(head, eye1, 0.05), eye2, 0.05);
}

float sdDragonWings(vec3 p) {
    // Wings as larger rotated cones
    vec3 wingBase = vec3(-0.2, 0.0, 0.0);
    float wing1 = sdRoundCone(rotZ(p - wingBase, -0.5), 0.0, 0.3, 0.7);
    float wing2 = sdRoundCone(rotZ(p - wingBase, 0.5), 0.0, 0.3, 0.7);
    return opSmoothUnion(wing1, wing2, 0.1);
}

float sdDragonLegs(vec3 p) {
    // Legs as capsules
    float leg1 = sdCapsule(p, vec3(-0.3, -0.8, 0.1), vec3(-0.3, -0.5, 0.1), 0.05);
    float leg2 = sdCapsule(p, vec3(0.3, -0.8, 0.1), vec3(0.3, -0.5, 0.1), 0.05);
    return opSmoothUnion(leg1, leg2, 0.1);
}

float sdf(vec3 p) {
    // Combine all parts of the dragon
    float body = sdDragonBody(p);
    float head = sdDragonHead(p);
    float wings = sdDragonWings(p);
    float legs = sdDragonLegs(p);
    return opSmoothUnion(opSmoothUnion(opSmoothUnion(body, head, 0.1), wings, 0.1), legs, 0.1);
}

vec3 sdf_color(vec3 p) {
    // Color the dragon with a gradient from green to red
    float height = p.y + 1.0;
    return mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), height / 2.0);
}
