#include "StdInc.h"
#include "SocketPolicyTransport.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <array>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace
{
void waitSocket(int fd, short events, native_session::Deadline deadline)
{
	while(true)
	{
		const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
		if(remaining <= 0) throw std::runtime_error("Policy socket timeout");
		pollfd descriptor{fd, events, 0};
		const int result = ::poll(&descriptor, 1, static_cast<int>(std::min<int64_t>(remaining, 300000)));
		if(result < 0 && errno == EINTR) continue;
		if(result == 0) continue;
		if(result < 0 || !(descriptor.revents & events))
			throw std::runtime_error(result == 0 ? "Policy socket timeout" : "Policy socket disconnected");
		return;
	}
}
/// A string as JsonWriter::writeString writes it, including a backslash it
/// takes for an escape already made.
void appendString(std::string & out, const std::string & text)
{
	// escaped[i] is written as a backslash and codes[i]; a table finds them, as escaped.find did
	static const std::string escaped = "\"\\\b\f\n\r\t";
	static const std::string codes = "\"\\bfnrt";
	static const auto tables = []
	{
		std::array<std::array<unsigned char, 256>, 2> made{};
		for(size_t i = 0; i < escaped.size(); ++i) made[0][(unsigned char)escaped[i]] = (unsigned char)(i + 1);
		for(size_t i = 0; i < codes.size(); ++i) made[1][(unsigned char)codes[i]] = 1;
		return made;
	}();
	const auto & escape = tables[0];
	const auto & isCode = tables[1];
	out += '"';
	size_t start = 0;
	const size_t size = text.size();
	for(size_t pos = 0; pos < size; ++pos)
	{
		const unsigned char found = escape[(unsigned char)text[pos]];
		if(!found)
			continue;
		if(text[pos] == '\\' && pos + 1 < size && isCode[(unsigned char)text[pos + 1]])
		{
			++pos; // such a pair stays as it is
			continue;
		}
		out.append(text, start, pos - start);
		out += '\\';
		out += codes[found - 1];
		start = pos + 1;
	}
	out.append(text, start, size - start);
	out += '"';
}

/// One line of JSON for the policy, appended to ``out``. JsonNode::toCompactString
/// writes the same values through a stream, pretty-printed with tabs and
/// spaces between them; this writes them alone, so a reader gets the same
/// values from fewer bytes. Returns false for a float, which is left to
/// JsonWriter's formatting.
bool appendJson(std::string & out, const JsonNode & node)
{
	switch(node.getType())
	{
	case JsonNode::JsonType::DATA_NULL:
		out += "null";
		return true;
	case JsonNode::JsonType::DATA_BOOL:
		out += node.Bool() ? "true" : "false";
		return true;
	case JsonNode::JsonType::DATA_FLOAT:
		return false;
	case JsonNode::JsonType::DATA_INTEGER:
	{
		char digits[24];
		out.append(digits, std::to_chars(digits, digits + sizeof(digits), node.Integer()).ptr);
		return true;
	}
	case JsonNode::JsonType::DATA_STRING:
	{
		const auto & text = node.String();
		static const size_t mark = std::strlen(explorer_rl_json::RAW_JSON_MARK);
		if(text.size() >= mark && text.compare(0, mark, explorer_rl_json::RAW_JSON_MARK) == 0)
			out.append(text, mark, std::string::npos);  // JSON written already
		else
			appendString(out, text);
		return true;
	}
	case JsonNode::JsonType::DATA_VECTOR:
	{
		out += '[';
		bool first = true;
		for(const auto & item : node.Vector())
		{
			if(!first) out += ',';
			first = false;
			if(!appendJson(out, item)) return false;
		}
		out += ']';
		return true;
	}
	case JsonNode::JsonType::DATA_STRUCT:
	{
		out += '{';
		bool first = true;
		for(const auto & [name, item] : node.Struct())
		{
			if(!first) out += ',';
			first = false;
			appendString(out, name);
			out += ':';
			if(!appendJson(out, item)) return false;
		}
		out += '}';
		return true;
	}
	}
	return false;
}

}  // namespace

const char explorer_rl_json::RAW_JSON_MARK[] = "\x01raw-json\x01";

bool explorer_rl_json::appendMerged(std::string & out, const JsonNode & first, const JsonNode & second)
{
	if(first.getType() != JsonNode::JsonType::DATA_STRUCT || (second.getType() != JsonNode::JsonType::DATA_STRUCT && !second.isNull()))
		return false;
	static const JsonMap empty;
	const auto & a = first.Struct();
	const auto & b = second.isNull() ? empty : second.Struct();
	out += '{';
	auto x = a.begin(), y = b.begin();
	bool firstItem = true;
	while(x != a.end() || y != b.end())
	{
		const bool fromFirst = y == b.end() || (x != a.end() && x->first < y->first);
		const auto & entry = fromFirst ? *x : *y;
		if(!firstItem) out += ',';
		firstItem = false;
		appendString(out, entry.first);
		out += ':';
		if(!appendJson(out, entry.second)) return false;
		if(fromFirst) ++x; else ++y;
	}
	out += '}';
	return true;
}

namespace
{
/// The policy's answer as JsonParser (strict JSON) reads it, for the answers it writes: objects of
/// ASCII string keys without escapes or '#', and values that are such strings, integers, null, true, false or
/// such objects, with no white space. Anything else returns false, and JsonParser reads the line. The nodes
/// are made as JsonParser makes them (extractStruct, extractString, extractFloat, extractNull, extractTrue,
/// extractFalse), with no override flag and no mod scope.
class AnswerReader
{
	const char * p;
	const char * end;
	int depth = 0;

	bool string(std::string & out)
	{
		if(p >= end || *p != '"')
			return false;
		const char * start = ++p;
		while(p < end && *p != '"')
		{
			const unsigned char c = static_cast<unsigned char>(*p);
			if(c < 0x20 || c >= 0x80 || c == '\\')
				return false;
			++p;
		}
		if(p >= end)
			return false;
		out.assign(start, p - start);
		++p;
		return true;
	}
	bool literal(const char * word, size_t size)
	{
		if(static_cast<size_t>(end - p) < size || std::memcmp(p, word, size) != 0)
			return false;
		p += size;
		return p >= end || *p == ',' || *p == '}';
	}
	bool value(JsonNode & node)
	{
		if(p >= end)
			return false;
		const char c = *p;
		if(c == '{')
			return object(node);
		if(c == '"')
		{
			std::string text;
			if(!string(text))
				return false;
			node.setType(JsonNode::JsonType::DATA_STRING);
			node.String() = text;
			return true;
		}
		if(c == 'n')
		{
			if(!literal("null", 4))
				return false;
			node.clear();
			return true;
		}
		if(c == 't' || c == 'f')
		{
			const bool truth = c == 't';
			if(!(truth ? literal("true", 4) : literal("false", 5)))
				return false;
			node.Bool() = truth;
			return true;
		}
		if(c == '-' || (c >= '0' && c <= '9'))
		{
			const bool negative = c == '-';
			if(negative)
				++p;
			const char * digits = p;
			int64_t number = 0;
			while(p < end && *p >= '0' && *p <= '9')
			{
				number = number * 10 + (*p - '0');
				++p;
			}
			const auto count = p - digits;
			if(count == 0 || count > 18 || (count > 1 && *digits == '0'))
				return false;
			if(p < end && *p != ',' && *p != '}')
				return false;
			node.setType(JsonNode::JsonType::DATA_INTEGER);
			node.Integer() = negative ? -number : number;
			return true;
		}
		return false;
	}
	bool object(JsonNode & node)
	{
		if(++depth > 8)
			return false;
		++p; // '{'
		node.setType(JsonNode::JsonType::DATA_STRUCT);
		if(p < end && *p == '}')
		{
			++p;
			--depth;
			return true;
		}
		while(true)
		{
			std::string key;
			if(!string(key) || key.find('#') != std::string::npos || node.Struct().count(key))
				return false;
			if(p >= end || *p != ':')
				return false;
			++p;
			if(!value(node.Struct()[key]))
				return false;
			if(p < end && *p == ',')
			{
				++p;
				continue;
			}
			if(p < end && *p == '}')
			{
				++p;
				--depth;
				return true;
			}
			return false;
		}
	}
public:
	AnswerReader(const char * data, size_t size) : p(data), end(data + size) {}
	bool read(JsonNode & node)
	{
		if(p >= end || *p != '{')
			return false;
		return object(node) && p == end;
	}
};

class SocketPolicy final : public native_session::PolicyTransport
{
	int fd = -1;
	int timeoutSeconds;
	std::string buffer;
	size_t lastLineBytes = 0;
	bool checkAnswers = false; ///< HEROES_CAPTURE_CHECK=1: every answer the fast reader read is read by JsonParser too
public:
	/// A TCP connection to the loopback ``port``, or with a ``path`` a connection to that local (Unix) socket.
	SocketPolicy(int port, const std::string & path, int seconds) : timeoutSeconds(seconds)
	{
		const char * check = std::getenv("HEROES_CAPTURE_CHECK");
		checkAnswers = check && std::string(check) == "1";
		fd = ::socket(path.empty() ? AF_INET : AF_UNIX, SOCK_STREAM, 0);
		if(fd < 0) throw std::runtime_error("Cannot create policy socket");
		try
		{
#ifdef SO_NOSIGPIPE
			const int yes = 1;
			::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes));
#endif
			timeval timeout{seconds, 0};
			::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
			::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
			if(path.empty())
			{
				const int noDelay = 1;
				::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &noDelay, sizeof(noDelay));
				sockaddr_in address{};
				address.sin_family = AF_INET;
				address.sin_port = htons(static_cast<uint16_t>(port));
				address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
				if(::connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0)
					throw std::runtime_error("Cannot connect to local policy server");
			}
			else
			{
				sockaddr_un address{};
				if(path.size() >= sizeof(address.sun_path))
					throw std::runtime_error("Policy socket path is too long");
				address.sun_family = AF_UNIX;
				std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
				if(::connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0)
					throw std::runtime_error("Cannot connect to local policy socket");
			}
		}
		catch(...) { ::close(fd); fd = -1; throw; }
	}
	~SocketPolicy() override { if(fd >= 0) ::close(fd); }
	void close() noexcept override { if(fd >= 0) ::shutdown(fd, SHUT_RDWR); }
	native_session::WriteMetrics send(const JsonNode & message) override
	{
		const auto start = std::chrono::steady_clock::now();
		std::string line;
		line.reserve(lastLineBytes + 256);
		if(!appendJson(line, message))
		{
			line = message.toCompactString();
			line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
			line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
			if(line.find(explorer_rl_json::RAW_JSON_MARK) != std::string::npos)
				throw std::runtime_error("A message with a float holds JSON written for the fast writer");
		}
		lastLineBytes = line.size();
		line += '\n';
		const auto encoded = std::chrono::steady_clock::now();
		const auto deadline = timeoutSeconds == 0 ? native_session::Deadline::max()
			: encoded + std::chrono::seconds(timeoutSeconds);
		size_t offset = 0;
		while(offset < line.size())
		{
#ifdef MSG_NOSIGNAL
			constexpr int flags = MSG_NOSIGNAL | MSG_DONTWAIT;
#else
			constexpr int flags = MSG_DONTWAIT;
#endif
			// Written at once when the socket has room (almost always); waited for only when it has none.
			const auto count = ::send(fd, line.data() + offset, line.size() - offset, flags);
			if(count < 0 && errno == EINTR) continue;
			if(count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
			{
				waitSocket(fd, POLLOUT, deadline);
				continue;
			}
			if(count <= 0) throw std::runtime_error("Cannot write policy socket");
			offset += static_cast<size_t>(count);
		}
		return {std::chrono::duration_cast<std::chrono::microseconds>(encoded - start).count(),
			std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - encoded).count(),
			static_cast<int64_t>(line.size())};
	}
	JsonNode receive(native_session::Deadline deadline) override
	{
		while(true)
		{
			const auto newline = buffer.find('\n');
			if(newline != std::string::npos)
			{
				if(newline > 65536) throw std::runtime_error("Policy response exceeds 64 KiB");
				const auto line = buffer.substr(0, newline);
				buffer.erase(0, newline + 1);
				JsonParsingSettings parsing;
				parsing.mode = JsonParsingSettings::JsonFormatMode::JSON;
				parsing.strict = true;
				JsonNode fast;
				if(AnswerReader(line.data(), line.size()).read(fast))
				{
					if(checkAnswers && !(fast == JsonNode(line.data(), line.size(), parsing, "ExplorerRL policy response")))
						throw std::runtime_error("Capture check: the fast answer reader differs from JsonParser");
					return fast;
				}
				return JsonNode(line.data(), line.size(), parsing, "ExplorerRL policy response");
			}
			if(buffer.size() > 65536) throw std::runtime_error("Policy response exceeds 64 KiB");
			waitSocket(fd, POLLIN, deadline);
			char bytes[4096];
			const auto count = ::recv(fd, bytes, sizeof(bytes), 0);
			if(count < 0 && errno == EINTR) continue;
			if(count <= 0) throw std::runtime_error("Policy socket disconnected");
			buffer.append(bytes, static_cast<size_t>(count));
		}
	}
};
}
std::shared_ptr<native_session::PolicyTransport> makeSocketPolicyTransport(int port, int timeoutSeconds)
{
	return std::make_shared<SocketPolicy>(port, std::string(), timeoutSeconds);
}

std::shared_ptr<native_session::PolicyTransport> makeLocalSocketPolicyTransport(const std::string & path, int timeoutSeconds)
{
	return std::make_shared<SocketPolicy>(0, path, timeoutSeconds);
}
