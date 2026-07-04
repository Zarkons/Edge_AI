#include "CameraInputHandler.h"
#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#include <cstring>

namespace obj_rec
{
    namespace app
    {
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
            std::cout << "[CameraInputHandler] Bypassing native VideoCapture. Opening direct socket stream..." << std::endl;

            std::string host, port, path;
            if (!ParseUrl(stream_url, host, port, path))
            {
                std::cerr << "[CameraInputHandler Error] Invalid URL layout passed." << std::endl;
                return false;
            }

            struct addrinfo hints{}, *res;
            hints.ai_family = AF_INET;
            hints.ai_socktype = SOCK_STREAM;

            if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0)
            {
                std::cerr << "[CameraInputHandler Error] Host resolution failed." << std::endl;
                return false;
            }

            m_sock_fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
            if (m_sock_fd < 0)
            {
                freeaddrinfo(res);
                return false;
            }

            if (::connect(m_sock_fd, res->ai_addr, res->ai_addrlen) < 0)
            {
                std::cerr << "[CameraInputHandler Error] Socket connection refused by iPhone." << std::endl;
                ::close(m_sock_fd);
                m_sock_fd = -1;
                freeaddrinfo(res);
                return false;
            }
            freeaddrinfo(res);

            // Send a standard raw HTTP GET request to pull the continuous MJPEG stream
            std::string request = "GET " + path + " HTTP/1.1\r\nHost: " + host + "\r\nConnection: close\r\n\r\n";
            ::send(m_sock_fd, request.c_str(), request.length(), 0);

            std::cout << "[CameraInputHandler] HTTP Network handshake complete. Receiving live stream bytes." << std::endl;
            return true;
        }

        bool CameraInputHandler::GetNextFrame(CameraFrame &out_frame)
        {
            if (m_sock_fd < 0)
            {
                return false;
            }

            uint8_t recv_buf[4096];

            while (true)
            {
                // Scan the stream loop until we find a complete JPEG frame chunk boundary markers
                if (m_stream_buffer.size() > 4)
                {
                    size_t start_idx = std::string::npos;
                    size_t end_idx = std::string::npos;

                    for (size_t i = 0; i < m_stream_buffer.size() - 1; ++i)
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

                    // Extract and decode the JPEG block payload via memory vector
                    if (start_idx != std::string::npos && end_idx != std::string::npos)
                    {
                        std::vector<uint8_t> jpeg_bytes(m_stream_buffer.begin() + start_idx, m_stream_buffer.begin() + end_idx);

                        // Erase processed data block from streaming ring buffer tracking list
                        m_stream_buffer.erase(m_stream_buffer.begin(), m_stream_buffer.begin() + end_idx);

                        // Decode using OpenCV's memory image decoder (bypasses FFmpeg network backends)
                        cv::Mat raw_mat = cv::imdecode(jpeg_bytes, cv::IMREAD_COLOR);
                        if (raw_mat.empty())
                        {
                            continue;
                        }

                        out_frame.width = raw_mat.cols;
                        out_frame.height = raw_mat.rows;
                        out_frame.channels = 3;
                        out_frame.stride = static_cast<int32_t>(raw_mat.step);

                        size_t total_bytes = out_frame.stride * out_frame.height;
                        out_frame.data.resize(total_bytes);
                        std::memcpy(out_frame.data.data(), raw_mat.data, total_bytes);
                        return true;
                    }
                }

                // Read the next chunk of network packets into memory
                ssize_t bytes_received = ::recv(m_sock_fd, recv_buf, sizeof(recv_buf), 0);
                if (bytes_received <= 0)
                {
                    return false; // Connection closed or dropped
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
