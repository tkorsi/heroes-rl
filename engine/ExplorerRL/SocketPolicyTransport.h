#pragma once
#include "../../nativesession/PolicyTransport.h"
/// Legacy loopback TCP backend. OS-specific descriptors stay behind the seam.
std::shared_ptr<native_session::PolicyTransport> makeSocketPolicyTransport(int port, int timeoutSeconds);
/// The same backend over a local (Unix) stream socket at ``path``: no TCP processing on either side.
std::shared_ptr<native_session::PolicyTransport> makeLocalSocketPolicyTransport(const std::string & path, int timeoutSeconds);

class JsonNode;
namespace explorer_rl_json
{
/// A string value that starts with this is JSON already written: the writer copies the rest as it is.
extern const char RAW_JSON_MARK[];
/// ``first`` and ``second``, two objects without a key in common, written as one object, keys in order as a
/// JsonNode holding both would have them; false (``out`` unusable) for a float, which the writer leaves to VCMI.
bool appendMerged(std::string & out, const JsonNode & first, const JsonNode & second);
}
