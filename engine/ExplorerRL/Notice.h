#pragma once

#include "../../lib/constants/EntityIdentifiers.h"
#include "../../lib/json/JsonNode.h"
#include "../../lib/networkPacks/Component.h"
#include "../../lib/networkPacks/EInfoWindowMode.h"
#include "../../lib/texts/MetaString.h"

#include <set>
#include <stdexcept>
#include <string>

namespace explorer_rl
{
/// A received information window outside the audited contextual baseline.
/// Diagnostics contain no received text, engine identifiers or hidden payload.
class UnsupportedNotice : public std::runtime_error
{
public:
	explicit UnsupportedNotice(const char * reason) : std::runtime_error(reason) {}
};

/// Validate the complete native template and all pools before returning the
/// learner's Notice JSON. No localization, unused substitutions, native IDs or
/// unrelated packet fields survive. The caller supplies only its own heroes'
/// name IDs; this parser does not query the world or alter game state.
JsonNode normalizeNotice(EInfoWindowMode mode, const MetaString & text,
	const std::vector<Component> & components, PlayerColor recipient,
	const std::set<std::string> & ownHeroNameTextIds);
}
