// Thin CPython binding of the in-process EngineSession. GPL-2.0-or-later.
// No game rules live here: it converts the already-sanitized player stream
// and responses, and releases the GIL while the engine works.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "StdInc.h"
#include "EngineSession.h"
#include "ResourceUsage.h"
#include <cmath>
#include <sstream>

namespace
{
using native_session::BoundaryToken;
using native_session::Deadline;
using native_session::EngineSession;

PyObject * engineError = nullptr;

/// Owns one reference; releases it on scope exit unless released.
class Ref
{
	PyObject * value;
public:
	explicit Ref(PyObject * object = nullptr) : value(object) {}
	~Ref() { Py_XDECREF(value); }
	Ref(const Ref &) = delete;
	Ref & operator=(const Ref &) = delete;
	PyObject * get() const { return value; }
	PyObject * release() { auto * result = value; value = nullptr; return result; }
	explicit operator bool() const { return value != nullptr; }
};

/// Run engine work without the GIL; C++ exceptions become Python exceptions.
template<class F> bool withoutGil(F && action)
{
	std::exception_ptr failure;
	Py_BEGIN_ALLOW_THREADS
	try { action(); }
	catch(...) { failure = std::current_exception(); }
	Py_END_ALLOW_THREADS
	if(!failure) return true;
	try { std::rethrow_exception(failure); }
	catch(const std::invalid_argument & error) { PyErr_SetString(PyExc_ValueError, error.what()); }
	catch(const std::exception & error) { PyErr_SetString(engineError, error.what()); }
	catch(...) { PyErr_SetString(engineError, "Unknown native engine failure"); }
	return false;
}

// --- engine stream to Python ------------------------------------------------
// The executable backend serializes each message with JsonWriter and the
// learner decodes that line with json.loads. These conversions produce the
// same Python values, without the text round trip on the common path.

PyObject * jsonModuleLoads()
{
	static PyObject * loads = nullptr;
	if(!loads)
	{
		Ref module(PyImport_ImportModule("json"));
		if(!module) return nullptr;
		loads = PyObject_GetAttrString(module.get(), "loads");
	}
	return loads;
}

/// Exactly JsonWriter's text for one string, decoded as json.loads would.
PyObject * stringValue(const std::string & text)
{
	bool plain = true;
	for(const unsigned char character : text)
		if(character == '\\' || character < 0x20) { plain = false; break; }
	if(plain) // quotes are escaped and decoded symmetrically; bytes pass unchanged
		return PyUnicode_DecodeUTF8(text.data(), static_cast<Py_ssize_t>(text.size()), "surrogatepass");
	// Rare: JsonWriter keeps already-escaped sequences and raw control bytes.
	const auto literal = JsonNode(text).toCompactString();
	auto * loads = jsonModuleLoads();
	if(!loads) return nullptr;
	Ref encoded(PyUnicode_DecodeUTF8(literal.data(), static_cast<Py_ssize_t>(literal.size()), "surrogatepass"));
	if(!encoded) return nullptr;
	return PyObject_CallOneArg(loads, encoded.get());
}

/// ostream formatting, as JsonWriter writes floats, then JSON number grammar.
PyObject * floatValue(double value)
{
	std::ostringstream stream;
	stream << value;
	const auto text = stream.str();
	if(!std::isfinite(value))
	{
		PyErr_Format(PyExc_ValueError, "Nonfinite JSON number: %s", text.c_str());
		return nullptr;
	}
	if(text.find_first_of(".eE") == std::string::npos)
		return PyLong_FromString(text.c_str(), nullptr, 10); // json.loads yields int
	const double parsed = PyOS_string_to_double(text.c_str(), nullptr, PyExc_ValueError);
	if(parsed == -1.0 && PyErr_Occurred()) return nullptr;
	return PyFloat_FromDouble(parsed);
}

/// The socket line drops JsonWriter's newlines, so a mod-scope comment that
/// JsonWriter writes before an entry of a non-compact container makes the
/// line undecodable. Refuse it the same way; messages normally carry none.
bool commentFree(const JsonNode & container, const JsonNode & entry)
{
	if(entry.getModScope().empty() || container.isCompact()) return true;
	PyErr_SetString(PyExc_ValueError, "Mod-scope comment would break the socket line");
	return false;
}

PyObject * toPython(const JsonNode & node)
{
	switch(node.getType())
	{
		case JsonNode::JsonType::DATA_NULL:
			Py_RETURN_NONE;
		case JsonNode::JsonType::DATA_BOOL:
			return PyBool_FromLong(node.Bool());
		case JsonNode::JsonType::DATA_INTEGER:
			return PyLong_FromLongLong(node.Integer());
		case JsonNode::JsonType::DATA_FLOAT:
			return floatValue(node.Float());
		case JsonNode::JsonType::DATA_STRING:
			return stringValue(node.String());
		case JsonNode::JsonType::DATA_VECTOR:
		{
			const auto & items = node.Vector();
			Ref list(PyList_New(static_cast<Py_ssize_t>(items.size())));
			if(!list) return nullptr;
			for(size_t index = 0; index < items.size(); ++index)
			{
				if(!commentFree(node, items[index])) return nullptr;
				auto * item = toPython(items[index]);
				if(!item) return nullptr;
				PyList_SET_ITEM(list.get(), static_cast<Py_ssize_t>(index), item);
			}
			return list.release();
		}
		case JsonNode::JsonType::DATA_STRUCT:
		{
			Ref dict(PyDict_New()); // std::map order is the serialized key order
			if(!dict) return nullptr;
			for(const auto & [key, value] : node.Struct())
			{
				if(!commentFree(node, value)) return nullptr;
				Ref name(stringValue(key));
				if(!name) return nullptr;
				Ref item(toPython(value));
				if(!item || PyDict_SetItem(dict.get(), name.get(), item.get()) != 0) return nullptr;
			}
			return dict.release();
		}
	}
	PyErr_SetString(engineError, "Unknown native JSON type");
	return nullptr;
}

/// Converts in order up to the first message json.loads would refuse, like a
/// socket reader that delivered the lines before an undecodable one. The
/// refusal is returned as text; the channel's messages are already taken.
PyObject * messageList(const std::vector<JsonNode> & messages, std::string & refused)
{
	Ref list(PyList_New(0));
	if(!list) return nullptr;
	for(const auto & message : messages)
	{
		Ref item(toPython(message));
		if(!item)
		{
			if(!PyErr_ExceptionMatches(PyExc_ValueError) && !PyErr_ExceptionMatches(engineError)) return nullptr;
			PyObject * type = nullptr;
			PyObject * value = nullptr;
			PyObject * traceback = nullptr;
			PyErr_Fetch(&type, &value, &traceback);
			Ref typeRef(type), valueRef(value), tracebackRef(traceback);
			Ref text(value ? PyObject_Str(value) : nullptr);
			const char * utf8 = text ? PyUnicode_AsUTF8(text.get()) : nullptr;
			refused = std::string(type ? reinterpret_cast<PyTypeObject *>(type)->tp_name : "Error") + ": " + (utf8 ? utf8 : "undecodable message");
			PyErr_Clear();
			break;
		}
		if(PyList_Append(list.get(), item.get()) != 0) return nullptr;
	}
	return list.release();
}

PyObject * optionalText(const std::string & text)
{
	if(text.empty()) return Py_NewRef(Py_None);
	return PyUnicode_DecodeUTF8(text.data(), static_cast<Py_ssize_t>(text.size()), "replace");
}

// --- Python response to the engine -----------------------------------------
// The same values json.dumps then the strict VCMI parser would produce.

bool toJson(PyObject * value, JsonNode & output, int depth = 0)
{
	if(depth > 64) { PyErr_SetString(PyExc_ValueError, "Policy response nesting is too deep"); return false; }
	if(value == Py_None) { output = JsonNode(); return true; }
	if(PyBool_Check(value)) { output.Bool() = value == Py_True; return true; }
	if(PyLong_Check(value))
	{
		int overflow = 0;
		const long long number = PyLong_AsLongLongAndOverflow(value, &overflow);
		if(overflow) { PyErr_SetString(PyExc_OverflowError, "Policy response integer exceeds 64 bits"); return false; }
		if(number == -1 && PyErr_Occurred()) return false;
		output.Integer() = number;
		return true;
	}
	if(PyFloat_Check(value))
	{
		const double number = PyFloat_AS_DOUBLE(value);
		if(!std::isfinite(number)) { PyErr_SetString(PyExc_ValueError, "Policy response has a nonfinite number"); return false; }
		output.Float() = number;
		return true;
	}
	if(PyUnicode_Check(value))
	{
		Py_ssize_t size = 0;
		const char * text = PyUnicode_AsUTF8AndSize(value, &size);
		if(!text) return false;
		output.String() = std::string(text, static_cast<size_t>(size));
		return true;
	}
	if(PyList_Check(value) || PyTuple_Check(value))
	{
		Ref sequence(PySequence_Fast(value, "Policy response sequence"));
		if(!sequence) return false;
		auto & items = output.Vector();
		items.clear();
		const auto count = PySequence_Fast_GET_SIZE(sequence.get());
		for(Py_ssize_t index = 0; index < count; ++index)
		{
			items.emplace_back();
			if(!toJson(PySequence_Fast_GET_ITEM(sequence.get(), index), items.back(), depth + 1)) return false;
		}
		return true;
	}
	if(PyDict_Check(value))
	{
		auto & fields = output.Struct();
		fields.clear();
		PyObject * key = nullptr;
		PyObject * item = nullptr;
		Py_ssize_t position = 0;
		while(PyDict_Next(value, &position, &key, &item))
		{
			if(!PyUnicode_Check(key)) { PyErr_SetString(PyExc_TypeError, "Policy response keys must be strings"); return false; }
			Py_ssize_t size = 0;
			const char * text = PyUnicode_AsUTF8AndSize(key, &size);
			if(!text) return false;
			if(!toJson(item, fields[std::string(text, static_cast<size_t>(size))], depth + 1)) return false;
		}
		return true;
	}
	PyErr_Format(PyExc_TypeError, "Unsupported policy response value: %s", Py_TYPE(value)->tp_name);
	return false;
}

Deadline deadlineAfter(double seconds)
{
	if(!(seconds >= 0) || seconds > 86400) throw std::invalid_argument("Timeout must be 0..86400 seconds");
	return std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(seconds));
}

PyObject * timingsDict(const native_session::SessionTimings & value)
{
	return Py_BuildValue("{s:d,s:d,s:d,s:d,s:d,s:K,s:K,s:K,s:d,s:K,s:d}",
		"load_seconds", value.loadSeconds, "reset_seconds", value.resetSeconds,
		"advance_seconds", value.advanceSeconds, "cleanup_seconds", value.cleanupSeconds,
		"finish_wait_seconds", value.finishWaitSeconds, "episodes", static_cast<unsigned long long>(value.episodes),
		"steps", static_cast<unsigned long long>(value.steps), "opponent_turns", static_cast<unsigned long long>(value.opponentTurns),
		"opponent_turn_seconds", value.opponentTurnSeconds, "battles", static_cast<unsigned long long>(value.battles),
		"battle_seconds", value.battleSeconds);
}

PyObject * opponentDict(const native_session_host::OpponentStats & value)
{
	return Py_BuildValue("{s:s,s:i,s:i,s:O,s:O,s:O,s:K,s:d,s:K,s:d}",
		"module", value.module.c_str(), "player", value.player, "difficulty", value.difficulty,
		"created", value.created ? Py_True : Py_False, "released", value.released ? Py_True : Py_False,
		"module_reset", value.moduleReset ? Py_True : Py_False, "turns", static_cast<unsigned long long>(value.turns),
		"turn_seconds", value.turnSeconds, "battles", static_cast<unsigned long long>(value.battles),
		"battle_seconds", value.battleSeconds);
}

// --- Engine type ---------------------------------------------------------------

struct EngineObject
{
	PyObject_HEAD
	EngineSession * session;
	bool busy; ///< one controller thread at a time; checked under the GIL
};

/// Rejects a concurrent call from another Python thread while the GIL is released.
class Exclusive
{
	EngineObject * self;
	bool owned = false;
public:
	explicit Exclusive(EngineObject * value) : self(value)
	{
		if(self->busy) PyErr_SetString(engineError, "Engine is in use by another thread");
		else owned = self->busy = true;
	}
	~Exclusive() { if(owned) self->busy = false; }
	explicit operator bool() const { return owned; }
};

int engineInit(EngineObject * self, PyObject * args, PyObject * kwargs)
{
	static const char * keywords[] = {"console_log", nullptr};
	int consoleLog = 0;
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|p", const_cast<char **>(keywords), &consoleLog)) return -1;
	if(self->session) { PyErr_SetString(engineError, "Engine is already initialized"); return -1; }
	Exclusive exclusive(self);
	if(!exclusive) return -1;
	native_session::SessionOptions options;
	options.consoleLog = consoleLog != 0;
	EngineSession * created = nullptr;
	if(!withoutGil([&] { created = new EngineSession(options); })) return -1;
	self->session = created;
	return 0;
}

void destroySession(EngineObject * self)
{
	auto * session = self->session;
	self->session = nullptr;
	if(!session) return;
	Py_BEGIN_ALLOW_THREADS
	delete session; // joins any live game, then releases loaded resources
	Py_END_ALLOW_THREADS
}

void engineDealloc(EngineObject * self)
{
	destroySession(self); // no other reference, so no concurrent call
	Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

EngineSession * requireSession(EngineObject * self)
{
	if(!self->session) PyErr_SetString(engineError, "Engine is closed");
	return self->session;
}

PyObject * engineLoad(EngineObject * self, PyObject *)
{
	Exclusive exclusive(self);
	if(!exclusive) return nullptr;
	auto * session = requireSession(self);
	if(!session || !withoutGil([&] { session->load(); })) return nullptr;
	Py_RETURN_NONE;
}

PyObject * engineReset(EngineObject * self, PyObject * args, PyObject * kwargs)
{
	static const char * keywords[] = {"episode", "map", "capture_directory", "seed", "learner", "policy_timeout", "log_path", nullptr};
	const char * episode = nullptr;
	const char * map = nullptr;
	const char * capture = nullptr;
	const char * log = "";
	int seed = 0, learner = -1, timeout = 30;
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|$sssiiis", const_cast<char **>(keywords),
		&episode, &map, &capture, &seed, &learner, &timeout, &log)) return nullptr;
	if(!episode || !map || !capture)
	{
		PyErr_SetString(PyExc_TypeError, "reset() requires episode, map, capture_directory, seed and learner");
		return nullptr;
	}
	Exclusive exclusive(self);
	if(!exclusive) return nullptr;
	auto * session = requireSession(self);
	if(!session) return nullptr;
	native_session::EpisodeConfig config;
	config.episode = episode;
	config.map = map;
	config.captureDirectory = capture;
	config.seed = seed;
	config.learner = learner;
	config.policyTimeoutSeconds = timeout;
	config.logPath = log;
	if(!withoutGil([&] { session->reset(config); })) return nullptr;
	Py_RETURN_NONE;
}

PyObject * engineTake(EngineObject * self, PyObject * args, PyObject * kwargs)
{
	static const char * keywords[] = {"timeout", "until_boundary", nullptr};
	double timeout = 0;
	int untilBoundary = 1;
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "d|p", const_cast<char **>(keywords), &timeout, &untilBoundary)) return nullptr;
	Exclusive exclusive(self);
	if(!exclusive) return nullptr;
	auto * session = requireSession(self);
	if(!session) return nullptr;
	native_session::Delivery delivery;
	if(!withoutGil([&] { delivery = session->take(deadlineAfter(timeout), untilBoundary != 0); })) return nullptr;
	std::string refused;
	Ref messages(messageList(delivery.messages, refused));
	Ref failure(optionalText(delivery.failure));
	Ref decode(optionalText(refused));
	if(!messages || !failure || !decode) return nullptr;
	return Py_BuildValue("{s:K,s:K,s:O,s:O,s:O,s:O,s:O,s:O,s:O}",
		"generation", static_cast<unsigned long long>(delivery.token.generation),
		"decision", static_cast<unsigned long long>(delivery.token.decision),
		"messages", messages.get(), "needs_action", delivery.needsAction ? Py_True : Py_False,
		"terminal", delivery.terminalObserved ? Py_True : Py_False, "closed", delivery.closed ? Py_True : Py_False,
		"timed_out", delivery.timedOut ? Py_True : Py_False, "failure", failure.get(), "decode_error", decode.get());
}

PyObject * engineSubmit(EngineObject * self, PyObject * args)
{
	unsigned long long generation = 0, decision = 0;
	PyObject * response = nullptr;
	if(!PyArg_ParseTuple(args, "KKO!", &generation, &decision, &PyDict_Type, &response)) return nullptr;
	Exclusive exclusive(self);
	if(!exclusive) return nullptr;
	auto * session = requireSession(self);
	if(!session) return nullptr;
	JsonNode value;
	if(!toJson(response, value)) return nullptr;
	if(!withoutGil([&] { session->submit(BoundaryToken{generation, decision}, value); })) return nullptr;
	Py_RETURN_NONE;
}

/// The socket backend's exact response codec: the json.dumps line, its 64 KiB
/// admission and VCMI's strict JSON parser, without the socket.
PyObject * engineSubmitText(EngineObject * self, PyObject * args)
{
	unsigned long long generation = 0, decision = 0;
	const char * text = nullptr;
	Py_ssize_t size = 0;
	if(!PyArg_ParseTuple(args, "KKs#", &generation, &decision, &text, &size)) return nullptr;
	Exclusive exclusive(self);
	if(!exclusive) return nullptr;
	auto * session = requireSession(self);
	if(!session) return nullptr;
	if(!withoutGil([&] {
		if(size > 65536) throw std::invalid_argument("Policy response exceeds 64 KiB");
		JsonNode value;
		try
		{
			JsonParsingSettings parsing;
			parsing.mode = JsonParsingSettings::JsonFormatMode::JSON;
			parsing.strict = true;
			value = JsonNode(text, static_cast<size_t>(size), parsing, "ExplorerRL policy response");
		}
		catch(const std::exception & error) { throw std::invalid_argument(std::string("Policy response is not strict JSON: ") + error.what()); }
		session->submit(BoundaryToken{generation, decision}, value, true);
	})) return nullptr;
	Py_RETURN_NONE;
}

PyObject * engineFinish(EngineObject * self, PyObject * args, PyObject * kwargs)
{
	static const char * keywords[] = {"natural", "timeout", nullptr};
	int natural = -1;
	double timeout = -1;
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|$pd", const_cast<char **>(keywords), &natural, &timeout)) return nullptr;
	if(natural < 0 || timeout < 0)
	{
		PyErr_SetString(PyExc_TypeError, "finish() requires natural and a nonnegative timeout");
		return nullptr;
	}
	Exclusive exclusive(self);
	if(!exclusive) return nullptr;
	auto * session = requireSession(self);
	if(!session) return nullptr;
	native_session::EpisodeReport report;
	if(!withoutGil([&] { report = session->finish(natural != 0, deadlineAfter(timeout)); })) return nullptr;
	std::string refused;
	Ref messages(messageList(report.messages, refused));
	Ref opponent(opponentDict(report.opponent));
	Ref exitCode(report.exitCode ? PyLong_FromLong(*report.exitCode) : Py_NewRef(Py_None));
	Ref channel(optionalText(report.channelFailure));
	Ref close(optionalText(report.closeError));
	Ref decode(optionalText(refused));
	if(!messages || !opponent || !exitCode || !channel || !close || !decode) return nullptr;
	return Py_BuildValue("{s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:d,s:d}",
		"exit_code", exitCode.get(), "natural", report.natural ? Py_True : Py_False,
		"host_stopped", report.hostStopped ? Py_True : Py_False,
		"shutdown_deadline", report.shutdownDeadline ? Py_True : Py_False,
		"close_clean", report.closeClean ? Py_True : Py_False, "poisoned", report.poisoned ? Py_True : Py_False,
		"experiment_failed", report.experimentFailed ? Py_True : Py_False, "channel_failure", channel.get(),
		"close_error", close.get(), "messages", messages.get(), "decode_error", decode.get(), "opponent", opponent.get(),
		"wait_seconds", report.waitSeconds, "close_seconds", report.closeSeconds);
}

PyObject * engineTimings(EngineObject * self, PyObject *)
{
	Exclusive exclusive(self);
	if(!exclusive) return nullptr;
	auto * session = requireSession(self);
	return session ? timingsDict(session->timings()) : nullptr;
}

PyObject * engineClose(EngineObject * self, PyObject *)
{
	Exclusive exclusive(self);
	if(!exclusive) return nullptr;
	destroySession(self);
	Py_RETURN_NONE;
}

PyObject * enginePoisoned(EngineObject * self, void *)
{
	return PyBool_FromLong(self->session && self->session->poisoned());
}

PyObject * engineActive(EngineObject * self, void *)
{
	return PyBool_FromLong(self->session && self->session->active());
}

PyObject * resources(PyObject *, PyObject *)
{
	return toPython(native_session_host::resourceSnapshot());
}

/// Fixture support: the exact response and stream conversions, no engine.
PyObject * jsonRoundtrip(PyObject *, PyObject * value)
{
	JsonNode node;
	if(!toJson(value, node)) return nullptr;
	return toPython(node);
}

PyObject * jsonText(PyObject *, PyObject * value)
{
	JsonNode node;
	if(!toJson(value, node)) return nullptr;
	const auto text = node.toCompactString();
	return PyUnicode_DecodeUTF8(text.data(), static_cast<Py_ssize_t>(text.size()), "surrogatepass");
}

PyObject * parsedText(PyObject *, PyObject * value)
{
	// The socket backend's response parser, then JsonWriter text, for comparison.
	Py_ssize_t size = 0;
	const char * text = PyUnicode_AsUTF8AndSize(value, &size);
	if(!text) return nullptr;
	std::string serialized;
	if(!withoutGil([&] {
		JsonParsingSettings parsing;
		parsing.mode = JsonParsingSettings::JsonFormatMode::JSON;
		parsing.strict = true;
		serialized = JsonNode(text, static_cast<size_t>(size), parsing, "ExplorerRL policy response").toCompactString();
	})) return nullptr;
	return PyUnicode_DecodeUTF8(serialized.data(), static_cast<Py_ssize_t>(serialized.size()), "surrogatepass");
}

/// Measurement fixture on recorded player messages: the socket backend's
/// sender-side JsonWriter line versus this module's direct conversion.
PyObject * codecSeconds(PyObject *, PyObject * args)
{
	PyObject * texts = nullptr;
	int repeat = 1;
	if(!PyArg_ParseTuple(args, "O!i", &PyList_Type, &texts, &repeat)) return nullptr;
	if(repeat < 1 || repeat > 10000) { PyErr_SetString(PyExc_ValueError, "Use 1..10000 repeats"); return nullptr; }
	std::vector<JsonNode> nodes;
	for(Py_ssize_t index = 0; index < PyList_GET_SIZE(texts); ++index)
	{
		Py_ssize_t size = 0;
		const char * text = PyUnicode_AsUTF8AndSize(PyList_GET_ITEM(texts, index), &size);
		if(!text) return nullptr;
		JsonParsingSettings parsing;
		parsing.mode = JsonParsingSettings::JsonFormatMode::JSON;
		parsing.strict = true;
		try { nodes.emplace_back(text, static_cast<size_t>(size), parsing, "recorded player message"); }
		catch(const std::exception & error) { PyErr_SetString(PyExc_ValueError, error.what()); return nullptr; }
	}
	using Clock = std::chrono::steady_clock;
	size_t bytes = 0;
	auto start = Clock::now();
	for(int round = 0; round < repeat; ++round)
		for(const auto & node : nodes)
		{
			std::string line = node.toCompactString();
			line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
			line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
			bytes += line.size() + 1;
		}
	const double writer = std::chrono::duration<double>(Clock::now() - start).count();
	start = Clock::now();
	for(int round = 0; round < repeat; ++round)
		for(const auto & node : nodes)
		{
			Ref value(toPython(node));
			if(!value) return nullptr;
		}
	const double direct = std::chrono::duration<double>(Clock::now() - start).count();
	return Py_BuildValue("{s:n,s:i,s:d,s:d,s:K}", "messages", static_cast<Py_ssize_t>(nodes.size()), "repeat", repeat,
		"jsonwriter_seconds", writer, "to_python_seconds", direct, "wire_bytes", static_cast<unsigned long long>(bytes / repeat));
}

PyObject * streamValue(PyObject *, PyObject * args)
{
	// (kind, payload): float/string/integer native values as the stream converts them.
	const char * kind = nullptr;
	PyObject * payload = nullptr;
	if(!PyArg_ParseTuple(args, "sO", &kind, &payload)) return nullptr;
	JsonNode node;
	const std::string type(kind);
	if(type == "float") node.Float() = PyFloat_AsDouble(payload);
	else if(type == "string" && PyBytes_Check(payload)) node.String() = std::string(PyBytes_AS_STRING(payload), static_cast<size_t>(PyBytes_GET_SIZE(payload)));
	else { PyErr_SetString(PyExc_ValueError, "Use ('float', number) or ('string', bytes)"); return nullptr; }
	if(PyErr_Occurred()) return nullptr;
	// (converted value or None, JsonWriter bytes, conversion refusal or None)
	const auto serialized = node.toCompactString();
	Ref text(PyBytes_FromStringAndSize(serialized.data(), static_cast<Py_ssize_t>(serialized.size())));
	if(!text) return nullptr;
	std::string refused;
	Ref converted(messageList({node}, refused));
	if(!converted) return nullptr;
	Ref value(PyList_GET_SIZE(converted.get()) ? Py_NewRef(PyList_GET_ITEM(converted.get(), 0)) : Py_NewRef(Py_None));
	Ref refusal(optionalText(refused));
	if(!refusal) return nullptr;
	return Py_BuildValue("(OOO)", value.get(), text.get(), refusal.get());
}

PyMethodDef engineMethods[] = {
	{"load", reinterpret_cast<PyCFunction>(engineLoad), METH_NOARGS, "Load immutable game resources once per process."},
	{"reset", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(engineReset)), METH_VARARGS | METH_KEYWORDS, "Start a new game."},
	{"take", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(engineTake)), METH_VARARGS | METH_KEYWORDS, "Receive new player messages."},
	{"submit", reinterpret_cast<PyCFunction>(engineSubmit), METH_VARARGS, "Answer the current decision with typed values."},
	{"submit_text", reinterpret_cast<PyCFunction>(engineSubmitText), METH_VARARGS, "Answer with the socket backend's JSON line."},
	{"finish", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(engineFinish)), METH_VARARGS | METH_KEYWORDS, "End and destroy the game."},
	{"timings", reinterpret_cast<PyCFunction>(engineTimings), METH_NOARGS, "Cumulative native timings."},
	{"close", reinterpret_cast<PyCFunction>(engineClose), METH_NOARGS, "Destroy the session and its resources."},
	{nullptr, nullptr, 0, nullptr}};

PyGetSetDef engineProperties[] = {
	{"poisoned", reinterpret_cast<getter>(enginePoisoned), nullptr, "A failed owner needs a new process.", nullptr},
	{"active", reinterpret_cast<getter>(engineActive), nullptr, "A game is running.", nullptr},
	{nullptr, nullptr, nullptr, nullptr, nullptr}};

PyTypeObject engineType = {PyVarObject_HEAD_INIT(nullptr, 0)};

PyMethodDef moduleMethods[] = {
	{"resources", resources, METH_NOARGS, "Process memory/thread/descriptor telemetry."},
	{"json_roundtrip", jsonRoundtrip, METH_O, "Fixture: response conversion then stream conversion."},
	{"json_text", jsonText, METH_O, "Fixture: response conversion then JsonWriter text."},
	{"parsed_text", parsedText, METH_O, "Fixture: the socket response parser then JsonWriter text."},
	{"codec_seconds", codecSeconds, METH_VARARGS, "Measurement: socket-line encoding vs direct conversion of recorded messages."},
	{"stream_value", streamValue, METH_VARARGS, "Fixture: (converted, JsonWriter bytes, refusal) for one native value."},
	{nullptr, nullptr, 0, nullptr}};

PyModuleDef moduleDefinition = {PyModuleDef_HEAD_INIT, "heroes_embedded_engine",
	"In-process VCMI game host for one learner seat against Nullkiller Easy.", -1, moduleMethods};
}

PyMODINIT_FUNC PyInit_heroes_embedded_engine()
{
	engineType.tp_name = "heroes_embedded_engine.Engine";
	engineType.tp_basicsize = sizeof(EngineObject);
	engineType.tp_flags = Py_TPFLAGS_DEFAULT;
	engineType.tp_doc = "One embedded engine per process: load once, then reset/take/submit/finish games.";
	engineType.tp_new = PyType_GenericNew;
	engineType.tp_init = reinterpret_cast<initproc>(engineInit);
	engineType.tp_dealloc = reinterpret_cast<destructor>(engineDealloc);
	engineType.tp_methods = engineMethods;
	engineType.tp_getset = engineProperties;
	if(PyType_Ready(&engineType) < 0) return nullptr;
	Ref module(PyModule_Create(&moduleDefinition));
	if(!module) return nullptr;
	engineError = PyErr_NewException("heroes_embedded_engine.EngineError", PyExc_RuntimeError, nullptr);
	if(!engineError || PyModule_AddObjectRef(module.get(), "EngineError", engineError) < 0) return nullptr;
	if(PyModule_AddObjectRef(module.get(), "Engine", reinterpret_cast<PyObject *>(&engineType)) < 0) return nullptr;
	if(PyModule_AddStringConstant(module.get(), "PROTOCOL", "heroes-embedded-engine-v1") < 0) return nullptr;
	return module.release();
}
