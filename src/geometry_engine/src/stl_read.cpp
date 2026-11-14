// STL reader (binary or ASCII) -> welded indexed mesh.
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "vibecad/printability.hpp"

namespace vibecad {

Mesh read_stl(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    const std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    std::vector<Vec3> soup;
    uint32_t count = 0;
    if (data.size() >= 84) std::memcpy(&count, data.data() + 80, 4);
    if (data.size() >= 84 && data.size() == 84 + size_t(count) * 50) {  // binary: size is exact
        soup.resize(size_t(count) * 3);
        for (uint32_t t = 0; t < count; ++t)
            std::memcpy(soup[size_t(t) * 3].data(), data.data() + 84 + size_t(t) * 50 + 12, 36);
    } else {  // ASCII: "vertex x y z" lines
        std::istringstream text(data);
        std::string word;
        while (text >> word)
            if (word == "vertex") {
                Vec3 v;
                text >> v[0] >> v[1] >> v[2];
                soup.push_back(v);
            }
        if (soup.empty() || soup.size() % 3) throw std::runtime_error(path + " is not a valid STL file");
    }
    return weld(soup);
}

}  // namespace vibecad
