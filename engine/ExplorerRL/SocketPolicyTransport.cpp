#include "StdInc.h"
#include "SocketPolicyTransport.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <cerrno>
#include <poll.h>
#include <sys/socket.h>
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
		if(result <= 0 || !(descriptor.revents & events))
			throw std::runtime_error(result == 0 ? "Policy socket timeout" : "Policy socket disconnected");
		return;
	}
}
class SocketPolicy final : public native_session::PolicyTransport
{
	int fd = -1;
	int timeoutSeconds;
	std::string buffer;
public:
	SocketPolicy(int port, int seconds) : timeoutSeconds(seconds)
	{
		fd = ::socket(AF_INET, SOCK_STREAM, 0);
		if(fd < 0) throw std::runtime_error("Cannot create policy socket");
		try
		{
#ifdef SO_NOSIGPIPE
			const int yes = 1;
			::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes));
#endif
			const int noDelay = 1;
			::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &noDelay, sizeof(noDelay));
			timeval timeout{seconds, 0};
			::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
			::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
			sockaddr_in address{};
			address.sin_family = AF_INET;
			address.sin_port = htons(static_cast<uint16_t>(port));
			address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
			if(::connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0)
				throw std::runtime_error("Cannot connect to local policy server");
		}
		catch(...) { ::close(fd); fd = -1; throw; }
	}
	~SocketPolicy() override { if(fd >= 0) ::close(fd); }
	void close() noexcept override { if(fd >= 0) ::shutdown(fd, SHUT_RDWR); }
	native_session::WriteMetrics send(const JsonNode & message) override
	{
		const auto start = std::chrono::steady_clock::now();
		std::string line = message.toCompactString();
		line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
		line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
		line += '\n';
		const auto encoded = std::chrono::steady_clock::now();
		const auto deadline = encoded + std::chrono::seconds(timeoutSeconds);
		size_t offset = 0;
		while(offset < line.size())
		{
			waitSocket(fd, POLLOUT, deadline);
#ifdef MSG_NOSIGNAL
			constexpr int flags = MSG_NOSIGNAL;
#else
			constexpr int flags = 0;
#endif
			const auto count = ::send(fd, line.data() + offset, line.size() - offset, flags);
			if(count < 0 && errno == EINTR) continue;
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
	return std::make_shared<SocketPolicy>(port, timeoutSeconds);
}
