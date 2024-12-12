#ifndef RC_VISION_CORE_LOGGER_HPP
#define RC_VISION_CORE_LOGGER_HPP

#include <string>
#include <iostream>
#include <mutex>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <queue>
#include <thread>
#include <condition_variable>
#include <atomic>

namespace rc_vision {
    namespace core {

        /**
         * @brief 日志记录类，提供多级别日志记录功能，支持异步写入和日志文件旋转。
         *
         * 该类采用单例模式，确保全局只有一个日志实例。支持将日志输出到控制台和文件，
         * 并根据设置的最大文件大小自动旋转日志文件。
         */
        class Logger {
        public:
            /**
             * @brief 日志级别枚举。
             *
             * 定义了四个日志级别：DEBUG, INFO, WARN, ERROR。
             */
            enum class LogLevel { DEBUG, INFO, WARN, ERROR };

            /**
             * @brief 获取 Logger 的单例实例。
             *
             * @return Logger& 单例实例的引用。
             */
            static Logger& getInstance();

            /**
             * @brief 设置当前日志级别。
             *
             * 只有级别高于或等于当前设置级别的日志消息会被记录。
             *
             * @param level 需要设置的日志级别。
             */
            void setLogLevel(LogLevel level);

            /**
             * @brief 设置日志文件输出，并指定最大文件大小。
             *
             * 如果设置了日志文件，日志消息将被写入该文件。超过最大文件大小时，
             * 日志文件将被旋转（重命名并新建文件）。
             *
             * @param file_path 日志文件的路径。
             * @param max_file_size 最大文件大小，单位为字节。默认为10MB。
             * @return true 如果日志文件设置成功。
             * @return false 如果无法打开日志文件。
             */
            bool setLogFile(const std::string& file_path, std::size_t max_file_size = 10 * 1024 * 1024); // 默认10MB

            /**
             * @brief 记录日志消息。
             *
             * 根据日志级别和当前设置，将日志消息输出到控制台和/或日志文件。
             * 该方法是线程安全的，并且日志消息会被异步写入。
             *
             * @param level 日志级别。
             * @param message 日志内容。
             * @param file 源文件名（自动填充）。
             * @param line 行号（自动填充）。
             * @param func 函数名（自动填充）。
             */
            void log(LogLevel level, const std::string& message,
                     const std::string& file = "", int line = 0, const std::string& func = "");

            /**
             * @brief 关闭日志系统。
             *
             * 该方法会终止日志写入线程，并确保所有待写入的日志消息被处理。
             */
            void shutdown();

        private:
            /**
             * @brief 默认构造函数。
             *
             * 初始化日志级别为 INFO，并启动日志处理线程。
             */
            Logger();

            /**
             * @brief 析构函数。
             *
             * 确保日志系统被正确关闭，释放相关资源。
             */
            ~Logger();

            // 禁用拷贝构造和赋值操作
            Logger(const Logger&) = delete;
            Logger& operator=(const Logger&) = delete;

            LogLevel current_level_; /**< 当前日志级别。 */
            std::mutex mtx_; /**< 互斥锁，用于保护日志级别的修改。 */

            // 日志文件相关
            std::ofstream log_file_; /**< 输出文件流，用于写入日志文件。 */
            std::string log_file_path_; /**< 当前日志文件路径。 */
            std::size_t max_file_size_; /**< 最大日志文件大小，超过时进行文件旋转。 */

            // 日志消息队列及线程相关
            std::queue<std::string> log_queue_; /**< 日志消息队列。 */
            std::mutex queue_mtx_; /**< 互斥锁，用于保护日志队列。 */
            std::condition_variable cv_; /**< 条件变量，用于通知日志线程有新消息。 */
            std::thread logging_thread_; /**< 日志处理线程。 */
            std::atomic<bool> exit_flag_; /**< 退出标志，指示日志线程是否应终止。 */

            /**
             * @brief 日志处理线程的方法。
             *
             * 不断从日志队列中取出消息，并将其写入控制台和日志文件。
             * 当收到退出信号时，线程会终止。
             */
            void processLogs();

            /**
             * @brief 获取当前时间的字符串表示。
             *
             * 格式为 "YYYY-MM-DD HH:MM:SS.mmm"。
             *
             * @return std::string 当前时间的字符串。
             */
            std::string getCurrentTime() const;

            /**
             * @brief 获取日志级别的字符串表示。
             *
             * @param level 日志级别。
             * @return std::string 日志级别的字符串。
             */
            std::string getLogLevelString(LogLevel level) const;

            /**
             * @brief 获取对应日志级别的颜色代码。
             *
             * 仅在支持 ANSI 颜色码的终端中有效。
             *
             * @param level 日志级别。
             * @return std::string 颜色代码。
             */
            std::string getColorCode(LogLevel level) const;

            /**
             * @brief 重置终端颜色。
             *
             * @return std::string 重置颜色的 ANSI 代码。
             */
            std::string resetColor() const;

            /**
             * @brief 检查日志文件大小，并在必要时执行日志文件旋转。
             *
             * 将当前日志文件重命名为带时间戳的备份文件，并重新创建新的日志文件。
             */
            void rotateLogFile();
        };

        /**
         * @brief 宏定义用于简化日志记录。
         *
         * 这些宏会自动填充源文件名、行号和函数名，方便调试和追踪日志来源。
         */
#define LOG_DEBUG(message) \
            rc_vision::core::Logger::getInstance().log(rc_vision::core::Logger::LogLevel::DEBUG, message, __FILE__, __LINE__, __func__)

#define LOG_INFO(message) \
            rc_vision::core::Logger::getInstance().log(rc_vision::core::Logger::LogLevel::INFO, message, __FILE__, __LINE__, __func__)

#define LOG_WARN(message) \
            rc_vision::core::Logger::getInstance().log(rc_vision::core::Logger::LogLevel::WARN, message, __FILE__, __LINE__, __func__)

#define LOG_ERROR(message) \
            rc_vision::core::Logger::getInstance().log(rc_vision::core::Logger::LogLevel::ERROR, message, __FILE__, __LINE__, __func__)

    } // namespace core
} // namespace rc_vision

#endif // RC_VISION_CORE_LOGGER_HPP
