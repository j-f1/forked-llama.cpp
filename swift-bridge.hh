//
//  swift-bridge.hpp
//  LLaMAcpp
//
//  Created by Jed Fox on 2023-03-12.
//

#ifndef swift_bridge_hpp
#define swift_bridge_hpp

#include <string>
#include <Foundation/Foundation.h>

// Only needed before Swift 5.9!
// Switch to the below and conversions will be baked into the language!
// -cxx-interoperability-mode=swift-5.9

void bridge_string(std::string s, void(^cb)(const char *str, int length)) {
    cb(s.c_str(), s.size());
}

std::string bridge_string(const char *s) {
    return s;
}

#endif /* swift_bridge_hpp */
