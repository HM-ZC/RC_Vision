/*
 * 提供日志记录功能，方便调试和运行时信息输出。
 */
#include "rc_vision/core/logger.hpp"

namespace rc_vision {
    namespace core {

        Logger& Logger::getInstance() {
            static Logger instance;
            return instance;
        }

        Logger::Logger()
                : current_level_(LogLevel::INFO),
                  max_file_size_(10 * 1024 * 1024), // 默认10MB
                  exit_flag_(false),
                  logging_thread_(&Logger::processLogs, this) {}

        Logger::~Logger() {
            shutdown();
        }

        void Logger::shutdown() {
            if (!exit_flag_) {
                exit_flag_ = true;
                cv_.notify_all();
                if (logging_thread_.joinable()) {
                    logging_thread_.join();
                }
                if (log_file_.is_open()) {
                    log_file_.close();
                }
            }
        }

        void Logger::setLogLevel(LogLevel level) {
            std::lock_guard<std::mutex> lock(mtx_);
            current_level_ = level;
        }

        bool Logger::setLogFile(const std::string& file_path, std::size_t max_file_size) {
            std::lock_guard<std::mutex> lock(mtx_);
            if (log_file_.is_open()) {
                log_file_.close();
            }
            log_file_path_ = file_path;
            max_file_size_ = max_file_size;
            log_file_.open(log_file_path_, std::ios::app);
            return log_file_.is_open();
        }

        void Logger::log(LogLevel level, const std::string& message,
                         const std::string& file, int line, const std::string& func) {
            if (level < current_level_) {
                return;
            }

            std::ostringstream oss;
            oss << "[" << getCurrentTime() << "] "
                << "[" << getLogLevelString(level) << "] ";

            // 添加文件名、行号和函数名信息
            if (!file.empty()) {
                oss << "[" << std::filesystem::path(file).filename().string()
                    << ":" << line << " (" << func << ")] ";
            }

            oss << message;

            std::string log_message = oss.str();

            // 将日志消息推入队列
            {
                std::lock_guard<std::mutex> lock(queue_mtx_);
                log_queue_.push(log_message);
            }
            cv_.notify_one();
        }

        void Logger::processLogs() {
            while (!exit_flag_) {
                std::unique_lock<std::mutex> lock(queue_mtx_);
                cv_.wait(lock, [this] { return !log_queue_.empty() || exit_flag_; });

                while (!log_queue_.empty()) {
                    std::string log_message = log_queue_.front();
                    log_queue_.pop();
                    lock.unlock();

                    // 输出到控制台
                    std::cout << getColorCode(current_level_) << log_message << resetColor() << std::endl;

                    // 输出到文件并检查是否需要旋转
                    if (log_file_.is_open()) {
                        log_file_ << log_message << std::endl;
                        log_file_.flush(); // 确保写入磁盘
                        rotateLogFile();
                    }

                    lock.lock();
                }
            }

            // 处理剩余的日志消息
            std::lock_guard<std::mutex> lock(queue_mtx_);
            while (!log_queue_.empty()) {
                std::string log_message = log_queue_.front();
                log_queue_.pop();

                // 输出到控制台
                std::cout << getColorCode(current_level_) << log_message << resetColor() << std::endl;

                // 输出到文件并检查是否需要旋转
                if (log_file_.is_open()) {
                    log_file_ << log_message << std::endl;
                    log_file_.flush(); // 确保写入磁盘
                    rotateLogFile();
                }
            }
        }

        std::string Logger::getCurrentTime() const {
            auto now = std::chrono::system_clock::now();
            auto in_time_t = std::chrono::system_clock::to_time_t(now);
            auto milliseconds =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                            now.time_since_epoch()) % 1000;

            std::ostringstream ss;
            ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X")
               << "." << std::setfill('0') << std::setw(3) << milliseconds.count();
            return ss.str();
        }

        std::string Logger::getLogLevelString(LogLevel level) const {
            switch(level) {
                case LogLevel::DEBUG: return "DEBUG";
                case LogLevel::INFO:  return "INFO ";
                case LogLevel::WARN:  return "WARN ";
                case LogLevel::ERROR: return "ERROR";
                default:              return "UNKWN";
            }
        }

        std::string Logger::getColorCode(LogLevel level) const {
            switch(level) {
                case LogLevel::DEBUG: return "\033[36m"; // Cyan
                case LogLevel::INFO:  return "\033[32m"; // Green
                case LogLevel::WARN:  return "\033[33m"; // Yellow
                case LogLevel::ERROR: return "\033[31m"; // Red
                default:              return "\033[0m";  // Reset
            }
        }

        std::string Logger::resetColor() const {
            return "\033[0m";
        }

        void Logger::rotateLogFile() {
            if (log_file_.tellp() >= static_cast<std::streampos>(max_file_size_)) {
                log_file_.close();

                // 生成备份文件名
                std::string backup_file = log_file_path_ + "." + getCurrentTime();
                // 替换空格和冒号以避免文件名问题
                for (auto& ch : backup_file) {
                    if (ch == ' ' || ch == ':') {
                        ch = '_';
                    }
                }

                // 重命名当前日志文件
                try {
                    std::filesystem::rename(log_file_path_, backup_file);
                } catch (const std::filesystem::filesystem_error& e) {
                    std::cerr << "Failed to rotate log file: " << e.what() << std::endl;
                }

                // 打开新的日志文件
                log_file_.open(log_file_path_, std::ios::app);
            }
        }

    } // namespace core
} // namespace rc_vision