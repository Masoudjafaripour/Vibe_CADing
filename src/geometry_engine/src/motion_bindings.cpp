// Python module `vibecad_motion` (pip install ./src/geometry_engine).
// src/animation/motion_fast.py wraps it with the stick-figure skeleton.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "vibecad/motion.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace vibecad::motion;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

Skeleton make_skeleton(const IntArray& parents, const Array& offsets, const std::vector<std::string>& names) {
    const int J = int(parents.size());
    if (offsets.ndim() != 2 || offsets.shape(0) != J || offsets.shape(1) != 3)
        throw py::value_error("offsets must have shape (J, 3)");
    Skeleton s;
    s.parents.assign(parents.data(), parents.data() + J);
    for (int j = 0; j < J; ++j) {
        s.offsets.push_back({offsets.at(j, 0), offsets.at(j, 1), offsets.at(j, 2)});
        if (s.parents[j] >= j) throw py::value_error("parents must come before children");
    }
    s.names = names.empty() ? std::vector<std::string>() : names;
    for (int j = int(s.names.size()); j < J; ++j) s.names.push_back("joint" + std::to_string(j));
    return s;
}

Clip clip_of(const Array& pos) {
    if (pos.ndim() != 3 || pos.shape(2) != 3) throw py::value_error("positions must have shape (T, J, 3)");
    return {pos.data(), int(pos.shape(0)), int(pos.shape(1))};
}

void check_motion(const Array& root, const Array& angles, int J) {
    if (root.ndim() != 2 || root.shape(1) != 3) throw py::value_error("root must have shape (T, 3)");
    if (angles.ndim() != 3 || angles.shape(0) != root.shape(0) || angles.shape(1) != J || angles.shape(2) != 3)
        throw py::value_error("angles must have shape (T, J, 3) matching root and skeleton");
}

}  // namespace

PYBIND11_MODULE(vibecad_motion, m) {
    m.doc() = "Skeletal motion: forward kinematics, foot locking (two-bone IK), metrics, BVH export.";

    m.def(
        "forward_kinematics",
        [](const Array& root, const Array& angles, const IntArray& parents, const Array& offsets) {
            const Skeleton s = make_skeleton(parents, offsets, {});
            check_motion(root, angles, s.size());
            Array pos({root.shape(0), py::ssize_t(s.size()), py::ssize_t(3)});
            forward_kinematics(s, root.data(), angles.data(), int(root.shape(0)), pos.mutable_data());
            return pos;
        },
        "root"_a, "angles"_a, "parents"_a, "offsets"_a, "root (T,3), XYZ Euler angles (T,J,3) -> positions (T,J,3).");

    m.def(
        "stance_toes",
        [](const Array& pos, int l_toe, int r_toe, double fps, double contact_height, double max_speed) {
            return stance_toes(clip_of(pos), l_toe, r_toe, fps, contact_height, max_speed);
        },
        "pos"_a, "l_toe"_a, "r_toe"_a, "fps"_a, "contact_height"_a = 0.03, "max_speed"_a = 1.0);

    m.def(
        "lock_feet",
        [](const Array& pos, std::array<int, 4> left, std::array<int, 4> right, double fps, double contact_height,
           double max_drift) {
            const Clip c = clip_of(pos);
            Array out({py::ssize_t(c.frames), py::ssize_t(c.joints), py::ssize_t(3)});
            lock_feet(c, {left[0], left[1], left[2], left[3]}, {right[0], right[1], right[2], right[3]}, fps,
                      contact_height, max_drift, out.mutable_data());
            return out;
        },
        "pos"_a, "left"_a, "right"_a, "fps"_a, "contact_height"_a = 0.03, "max_drift"_a = 0.35,
        "Two-bone IK foot locking; legs are (hip, knee, ankle, toe) joint indices.");

    m.def(
        "foot_sliding",
        [](const Array& pos, int l_toe, int r_toe, double fps, double contact_height) {
            return foot_sliding(clip_of(pos), l_toe, r_toe, fps, contact_height);
        },
        "pos"_a, "l_toe"_a, "r_toe"_a, "fps"_a, "contact_height"_a = 0.02);

    m.def(
        "stance_sliding",
        [](const Array& pos, const Array& reference, int l_toe, int r_toe, double fps) {
            return stance_sliding(clip_of(pos), clip_of(reference), l_toe, r_toe, fps);
        },
        "pos"_a, "reference"_a, "l_toe"_a, "r_toe"_a, "fps"_a);

    m.def(
        "jerk", [](const Array& pos, double fps) { return jerk(clip_of(pos), fps); }, "pos"_a, "fps"_a);

    m.def(
        "write_bvh",
        [](const std::string& path, const Array& root, const Array& angles, const IntArray& parents,
           const Array& offsets, const std::vector<std::string>& names, double fps) {
            const Skeleton s = make_skeleton(parents, offsets, names);
            check_motion(root, angles, s.size());
            write_bvh(path, s, root.data(), angles.data(), int(root.shape(0)), fps);
        },
        "path"_a, "root"_a, "angles"_a, "parents"_a, "offsets"_a, "names"_a, "fps"_a);
}
