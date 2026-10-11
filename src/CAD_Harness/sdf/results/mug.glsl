float sphere(vec3 p, float r) {
    return length(p) - r;
}

float box(vec3 p, vec3 b) {
    vec3 q = abs(p);
    return max(q.x - b.x, max(q.y - b.y, q.z - b.z));
}

float cylinder(vec3 p, float r, float h) {
    vec3 q = abs(p);
    float d = max(q.x - r, q.z - h);
    return max(q.y - h, d);
}

float torus(vec3 p, float R, float r) {
    vec2 q = vec2(length(p.xy), p.z);
    return length(q - vec2(R, 0.0)) - r;
}

float sdf(vec3 p) {
    // Mug body
    float body = sphere(p, 0.5);
    
    // Handle
    vec3 handle = vec3(0.0, 0.3, 0.3);
    float handle_sdf = cylinder(p - handle, 0.1, 0.2);
    float handle_offset = 0.1;
    float handle_sdf_offset = cylinder(p - handle - vec3(0.0, 0.0, handle_offset), 0.1, 0.2);
    float handle_sdf_final = max(handle_sdf, handle_sdf_offset);
    
    // Combine body and handle with union (min)
    float mug = min(body, handle_sdf_final);
    
    // Add a small indentation on the handle for realism
    float handle_indent = cylinder(p - handle - vec3(0.0, 0.0, handle_offset * 0.5), 0.05, 0.1);
    float handle_final = max(handle_sdf_final, handle_indent);
    
    // Final shape
    return mug;
}