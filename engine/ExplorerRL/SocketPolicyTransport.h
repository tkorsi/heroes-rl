#pragma once
#include "../../nativesession/PolicyTransport.h"
/// Legacy loopback TCP backend. OS-specific descriptors stay behind the seam.
std::shared_ptr<native_session::PolicyTransport> makeSocketPolicyTransport(int port, int timeoutSeconds);
