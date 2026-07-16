// Minimal stand-in for boost::algorithm::split / boost::is_any_of, providing
// exactly the behavior CSVReader.h relies on (default token_compress_off:
// empty tokens are kept). Lets CSVReader.h compile unmodified without boost.
#pragma once

#include <string>
#include <vector>

namespace boost {

inline std::string is_any_of(const std::string &delims) { return delims; }

namespace algorithm {

inline void split(std::vector<std::string> &out, const std::string &s,
                  const std::string &delims) {
  out.clear();
  std::string cur;
  for (char c : s) {
    if (delims.find(c) != std::string::npos) {
      out.push_back(cur);
      cur.clear();
    } else {
      cur += c;
    }
  }
  out.push_back(cur);
}

} // namespace algorithm
} // namespace boost
