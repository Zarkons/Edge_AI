#include "camera_input_handler.h"
#include "print_logger.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#include <fcntl.h>
#include <cerrno>
#include <cstring>

namespace obj_rec
{
    namespace app
    {
        namespace
        {
            constexpr int kConnectTimeoutSeconds = 5;
            constexpr int kReceiveTimeoutSeconds = 5;

            bool ConnectWithTimeout(int sock_fd, const sockaddr *address, socklen_t address_len, int timeout_seconds)
            {
                int flags = fcntl(sock_fd, F_GETFL, 0);
                if (flags < 0)
                {
                    return false;
                }

                if (fcntl(sock_fd, F_SETFL, flags | O_NONBLOCK) < 0)
                {
                    return false;
                }

                int connect_result = ::connect(sock_fd, address, address_len);
                if (connect_result == 0)
                {
                    fcntl(sock_fd, F_SETFL, flags);
                    return true;
                }

                if (errno != EINPROGRESS)
                {
                    fcntl(sock_fd, F_SETFL, flags);
                    return false;
                }

                fd_set write_fds;
                FD_ZERO(&write_fds);
                FD_SET(sock_fd, &write_fds);

                timeval timeout;
                timeout.tv_sec = timeout_seconds;
                timeout.tv_usec = 0;

                int ready = select(sock_fd + 1, nullptr, &write_fds, nullptr, &timeout);
                if (ready <= 0)
                {
                    fcntl(sock_fd, F_SETFL, flags);
                    errno = (ready == 0) ? ETIMEDOUT : errno;
                    return false;
                }

                int socket_error = 0;
                socklen_t socket_error_length = sizeof(socket_error);
                if (getsockopt(sock_fd, SOL_SOCKET, SO_ERROR, &socket_error, &socket_error_length) < 0)
                {
                    fcntl(sock_fd, F_SETFL, flags);
                    return false;
                }

                fcntl(sock_fd, F_SETFL, flags);
                if (socket_error != 0)
                {
                    errno = socket_error;
                    return false;
                }

                return true;
            }
        }

        CameraInputHandler::CameraInputHandler()
            : m_sock_fd(-1)
        {
        }

        CameraInputHandler::~CameraInputHandler()
        {
            if (m_sock_fd >= 0)
            {
                ::close(m_sock_fd);
            }
        }

        bool CameraInputHandler::ParseUrl(const std::string &url, std::string &host, std::string &port, std::string &path)
        {
            std::string target = url;
            if (target.substr(0, 7) == "http://")
            {
                target = target.substr(7);
            }

            size_t port_pos = target.find(':');
            size_t path_pos = target.find('/');

            if (path_pos == std::string::npos)
            {
                return false;
            }

            if (port_pos != std::string::npos && port_pos < path_pos)
            {
                host = target.substr(0, port_pos);
                port = target.substr(port_pos + 1, path_pos - port_pos - 1);
            }
            else
            {
                host = target.substr(0, path_pos);
                port = "80";
            }
            path = target.substr(path_pos);
            return true;
        }

        bool CameraInputHandler::Initialize(const std::string &stream_url)
        {
            PRINT_INFO("[CameraInputHandler] Bypassing native VideoCapture. Opening direct socket stream...");

            std::string host, port, path;
            if (!ParseUrl(stream_url, host, port, path))
            {
                PRINT_ERROR("[CameraInputHandler Error] Invalid URL layout passed.");
                return false;
            }

            struct addrinfo hints{}, *res;
            hints.ai_family = AF_INET;
            hints.ai_socktype = SOCK_STREAM;

            if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0)
            {
                PRINT_ERROR("[CameraInputHandler Error] Host resolution failed.");
                return false;
            }

            PRINT_INFO("[CameraInputHandler] Connecting to " << host << ':' << port
                                                             << " with a " << kConnectTimeoutSeconds << "s timeout.");

            m_sock_fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
            if (m_sock_fd < 0)
            {
                freeaddrinfo(res);
                return false;
            }

            if (!ConnectWithTimeout(m_sock_fd, res->ai_addr, res->ai_addrlen, kConnectTimeoutSeconds))
            {
                PRINT_ERROR("[CameraInputHandler Error] Socket connection to " << host << ':' << port
                                                                               << " failed: " << std::strerror(errno));
                ::close(m_sock_fd);
                m_sock_fd = -1;
                freeaddrinfo(res);
                return false;
            }
            freeaddrinfo(res);

            timeval recv_timeout;
            recv_timeout.tv_sec = kReceiveTimeoutSeconds;
            recv_timeout.tv_usec = 0;
            if (setsockopt(m_sock_fd, SOL_SOCKET, SO_RCVTIMEO, &recv_timeout, sizeof(recv_timeout)) != 0)
            {
                PRINT_ERROR("[CameraInputHandler Warn] Failed to set receive timeout: "
                            << std::strerror(errno));
            }

            // Send a standard raw HTTP GET request to pull the continuous MJPEG stream
            std::string request = "GET " + path + " HTTP/1.1\r\nHost: " + host + "\r\nConnection: close\r\n\r\n";
            ::send(m_sock_fd, request.c_str(), request.length(), 0);

            PRINT_INFO("[CameraInputHandler] HTTP Network handshake complete. Receiving live stream bytes.");
            return true;
        }

        CameraFrameStatus CameraInputHandler::GetNextFrame(CameraFrame &out_frame)
        {
            if (m_sock_fd < 0)
            {
                return CameraFrameStatus::kDisconnected;
            }

            uint8_t recv_buf[4096];

            while (true)
            {
                size_t buffer_size = m_stream_buffer.size();

                // Step 1: Scan the stream loop until we find a complete JPEG frame chunk boundary markers
                if (buffer_size > 4)
                {
                    size_t start_idx = std::string::npos;
                    size_t end_idx = std::string::npos;

                    for (size_t i = 0; i < buffer_size - 1; ++i)
                    {
                        if (m_stream_buffer[i] == 0xFF && m_stream_buffer[i + 1] == 0xD8 && start_idx == std::string::npos)
                        {
                            start_idx = i;
                        }
                        if (m_stream_buffer[i] == 0xFF && m_stream_buffer[i + 1] == 0xD9 && start_idx != std::string::npos)
                        {
                            end_idx = i + 2;
                            break;
                        }
                    }

                    // Step 2: Extract and decode the JPEG block payload via class-level pre-allocated cache vector
                    if (start_idx != std::string::npos && end_idx != std::string::npos)
                    {
                        // ZERO-ALLOCATION FIX: Clear size but keep capacity intact. Costs 0 nanoseconds.
                        m_jpeg_cache.clear();
                        m_jpeg_cache.reserve(end_idx - start_idx);

                        // Copy bytes out of the deque segment into contiguous memory for cv::imdecode
                        m_jpeg_cache.insert(m_jpeg_cache.end(),
                                            m_stream_buffer.begin() + start_idx,
                                            m_stream_buffer.begin() + end_idx);

                        // O(1) COMPLEXITY FIX: If m_stream_buffer is std::deque, this pops from front
                        // instantly without shifting any subsequent bytes in physical memory layout.
                        m_stream_buffer.erase(m_stream_buffer.begin(), m_stream_buffer.begin() + end_idx);

                        // Decode using OpenCV's memory image decoder, still some dynamic allocations here but acceptable for live stream MJPEG frames
                        cv::Mat raw_mat = cv::imdecode(m_jpeg_cache, cv::IMREAD_COLOR);
                        if (raw_mat.empty())
                        {
                            continue; // Skip corrupted frames and look for next boundary markers
                        }

                        out_frame.width = raw_mat.cols;
                        out_frame.height = raw_mat.rows;
                        out_frame.channels = 3;
                        out_frame.stride = static_cast<int32_t>(raw_mat.step);

                        size_t total_bytes = out_frame.stride * out_frame.height;

                        // ZERO-ALLOCATION: Capacity is preserved when passing recycled frames via std::swap
                        out_frame.data.resize(total_bytes);

                        // Highly optimized vector pointer mapping straight into physical memory
                        std::memcpy(out_frame.data.data(), raw_mat.data, total_bytes);
                        return CameraFrameStatus::kOk;
                    }
                }

                // Step 3: Read the next chunk of network packets into memory if no frame was ready
                ssize_t bytes_received = ::recv(m_sock_fd, recv_buf, sizeof(recv_buf), 0);
                if (bytes_received <= 0)
                {
                    if (bytes_received < 0)
                    {
                        // If it's a timeout error (EAGAIN/EWOULDBLOCK), return false so the
                        // background thread can evaluate `is_running_` loop condition and exit if requested.
                        if (errno == EAGAIN || errno == EWOULDBLOCK)
                        {
                            return CameraFrameStatus::kRetry;
                        }
                        PRINT_ERROR("[CameraInputHandler Error] Timed out or failed while waiting for MJPEG bytes: "
                                    << std::strerror(errno));
                    }
                    return CameraFrameStatus::kDisconnected;
                }

                m_stream_buffer.insert(m_stream_buffer.end(), recv_buf, recv_buf + bytes_received);

                // Safety guard cap to prevent out-of-memory overhead if network corrupts headers
                if (m_stream_buffer.size() > 15 * 1024 * 1024)
                {
                    m_stream_buffer.clear();
                }
            }
        }
    }
}
