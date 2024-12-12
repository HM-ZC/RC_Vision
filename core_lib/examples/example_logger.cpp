#include "rc_vision/core/logger.hpp"

using namespace rc_vision::core;

int main() {
    // 获取 Logger 实例
    Logger& logger = Logger::getInstance();

    // 加载日志配置
    if (!logger.loadConfig("logger_config.yaml")) {
        std::cerr << "Failed to load logger configuration." << std::endl;
    }

    // 记录不同级别的日志
    LOG_INFO("程序开始运行");
    LOG_DEBUG("这是一条调试信息");
    LOG_WARN("这是一条警告信息");
    LOG_ERROR("这是一条错误信息");

    // 模拟日志文件旋转
    for(int i = 0; i < 100000; ++i) {
        LOG_INFO("日志消息 " + std::to_string(i));
    }

    // 关闭日志系统
    logger.shutdown();

    return 0;
}