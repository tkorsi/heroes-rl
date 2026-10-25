#pragma once

#include "../../lib/json/JsonNode.h"
#include "../../lib/networkPacks/Component.h"
#include "../../lib/texts/MetaString.h"

#include <stdexcept>
#include <vector>

namespace explorer_rl
{
class UnsupportedDialog : public std::runtime_error
{
public:
	explicit UnsupportedDialog(const char * reason) : std::runtime_error(reason) {}
};

struct NormalizedDialog
{
	/// Only delivered display fields and semantic choices. The caller adds
	/// fresh opaque refs; native answer values never enter the learner JSON.
	JsonNode panel;
	std::vector<int> replies;
};

NormalizedDialog normalizeDialog(const MetaString & text,
	const std::vector<Component> & components, bool selection, bool cancel);
}
