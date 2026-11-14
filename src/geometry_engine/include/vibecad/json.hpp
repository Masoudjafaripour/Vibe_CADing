// Minimal JSON reader (objects, arrays, numbers, strings, true/false/null) for
// LLM-written part specs. Errors carry the byte offset so they can be fed back.
#pragma once
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace vibecad::json {

struct Value {
    enum class Type { Null, Bool, Number, String, Array, Object } type = Type::Null;
    bool boolean = false;
    double number = 0;
    std::string string;
    std::vector<Value> array;
    std::map<std::string, Value> object;

    bool is_object() const { return type == Type::Object; }
    bool is_array() const { return type == Type::Array; }
    bool is_number() const { return type == Type::Number; }
    bool is_string() const { return type == Type::String; }
    // Member lookup; nullptr when absent (or this is not an object).
    const Value* get(const std::string& key) const;
};

struct ParseError : std::runtime_error {
    size_t offset;
    ParseError(const std::string& what, size_t at) : std::runtime_error(what), offset(at) {}
};

// Parses the first JSON value in `text` (ignoring a ```json fence around it).
Value parse(const std::string& text);

}  // namespace vibecad::json
