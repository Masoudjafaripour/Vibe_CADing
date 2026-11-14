// Skeletal motion in C++: forward kinematics, foot contact, two-bone IK foot
// locking, motion metrics and BVH export. C++ twin of src/animation/
// (skeleton.py, metrics.py, control.py), which stays the reference.
#pragma once
#include <array>
#include <string>
#include <vector>

namespace vibecad::motion {

using Vec3 = std::array<double, 3>;

struct Skeleton {
    std::vector<std::string> names;
    std::vector<int> parents;  // -1 for the root; parents come before children
    std::vector<Vec3> offsets; // rest-pose offset from the parent joint

    int size() const { return int(parents.size()); }
};

// Row-major clip data: frames x joints x 3, as NumPy lays out a (T, J, 3) array.
struct Clip {
    const double* data;
    int frames, joints;
    const double* at(int t, int j) const { return data + (size_t(t) * joints + j) * 3; }
};

struct Leg {
    int hip, knee, ankle, toe;
};

// root (T, 3) + per-joint XYZ Euler angles (T, J, 3), R = Rx * Ry * Rz -> positions (T, J, 3).
void forward_kinematics(const Skeleton& skel, const double* root, const double* angles, int frames, double* pos);

// Per frame: index of the planted toe joint, or -1 (slower toe, on the ground, below max_speed).
std::vector<int> stance_toes(const Clip& pos, int l_toe, int r_toe, double fps, double contact_height,
                             double max_speed);

// Pins planted feet and re-solves each knee with two-bone IK; writes a corrected copy to `out`.
void lock_feet(const Clip& pos, const Leg& left, const Leg& right, double fps, double contact_height,
               double max_drift, double* out);

double foot_sliding(const Clip& pos, int l_toe, int r_toe, double fps, double contact_height);
double stance_sliding(const Clip& pos, const Clip& reference, int l_toe, int r_toe, double fps);
double jerk(const Clip& pos, double fps);

// BVH (Biovision hierarchy): opens in Blender, Maya, MotionBuilder, Unity.
void write_bvh(const std::string& path, const Skeleton& skel, const double* root, const double* angles, int frames,
               double fps);

}  // namespace vibecad::motion
