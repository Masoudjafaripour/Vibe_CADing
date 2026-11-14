#include "vibecad/json.hpp"

#include <cstdlib>

namespace vibecad::json {

const Value* Value::get(const std::string& key) const {
    if (type != Type::Object) return nullptr;
    auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
}

namespace {

class Parser {
public:
    explicit Parser(const std::string& s) : s_(s) {}

    Value document() {
        // Small models sometimes wrap the JSON in prose or a code fence: start at the first '{' or '['.
        const size_t start = s_.find_first_of("{[");
        if (start == std::string::npos) fail("no JSON object found");
        i_ = start;
        return value(0);
    }

private:
    const std::string& s_;
    size_t i_ = 0;

    [[noreturn]] void fail(const std::string& what) { throw ParseError("invalid JSON: " + what, i_); }

    void ws() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t')) ++i_;
    }
    char peek() {
        ws();
        if (i_ >= s_.size()) fail("unexpected end of input");
        return s_[i_];
    }
    void expect(char c) {
        if (peek() != c) fail(std::string("expected '") + c + "'");
        ++i_;
    }
    bool literal(const char* word) {
        const size_t n = std::char_traits<char>::length(word);
        if (s_.compare(i_, n, word) == 0) return i_ += n, true;
        return false;
    }

    Value value(int depth) {
        if (depth > 32) fail("nested too deeply");
        Value v;
        const char c = peek();
        if (c == '{') {
            v.type = Value::Type::Object;
            ++i_;
            if (peek() == '}') return ++i_, v;
            while (true) {
                if (peek() != '"') fail("expected a quoted key");
                std::string key = str();
                expect(':');
                v.object[key] = value(depth + 1);
                const char d = peek();
                ++i_;
                if (d == '}') break;
                if (d != ',') fail("expected ',' or '}'");
            }
        } else if (c == '[') {
            v.type = Value::Type::Array;
            ++i_;
            if (peek() == ']') return ++i_, v;
            while (true) {
                v.array.push_back(value(depth + 1));
                const char d = peek();
                ++i_;
                if (d == ']') break;
                if (d != ',') fail("expected ',' or ']'");
            }
        } else if (c == '"') {
            v.type = Value::Type::String;
            v.string = str();
        } else if (literal("true")) {
            v.type = Value::Type::Bool, v.boolean = true;
        } else if (literal("false")) {
            v.type = Value::Type::Bool;
        } else if (literal("null")) {
        } else if (c == '-' || (c >= '0' && c <= '9')) {
            char* end = nullptr;
            v.number = std::strtod(s_.c_str() + i_, &end);
            if (end == s_.c_str() + i_) fail("bad number");
            v.type = Value::Type::Number;
            i_ = size_t(end - s_.c_str());
        } else {
            fail(std::string("unexpected character '") + c + "'");
        }
        return v;
    }

    std::string str() {
        ++i_;  // opening quote
        std::string out;
        while (true) {
            if (i_ >= s_.size()) fail("unterminated string");
            const char c = s_[i_++];
            if (c == '"') break;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (i_ >= s_.size()) fail("unterminated escape");
            const char e = s_[i_++];
            switch (e) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'u':  // keep ASCII, replace anything else
                    if (i_ + 4 > s_.size()) fail("bad \\u escape");
                    {
                        const long code = std::strtol(s_.substr(i_, 4).c_str(), nullptr, 16);
                        out += code < 128 ? char(code) : '?';
                    }
                    i_ += 4;
                    break;
                default: out += e;  // \" \\ \/
            }
        }
        return out;
    }
};

}  // namespace

Value parse(const std::string& text) { return Parser(text).document(); }

}  // namespace vibecad::json
