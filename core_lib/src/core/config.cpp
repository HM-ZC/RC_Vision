/*
 * 处理配置管理，如读取和解析配置文件（例如 YAML、JSON），并提供访问配置参数的接口。
 */
#include "rc_vision/core/config.hpp"
#include <fstream>
#include <algorithm>

namespace rc_vision {
    namespace core {

        bool Config::loadYAML(const std::string& file_path) {
            try {
                config_yaml_ = YAML::LoadFile(file_path);
                Logger::getInstance().log(Logger::LogLevel::INFO, "Configuration loaded from YAML file: " + file_path);
                return true;
            } catch (const YAML::Exception& e) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, std::string("Failed to load YAML config file: ") + e.what());
                return false;
            }
        }

        bool Config::loadJSON(const std::string& file_path) {
            try {
                std::ifstream file(file_path);
                if(!file.is_open()) {
                    Logger::getInstance().log(Logger::LogLevel::ERROR, "Failed to open JSON config file: " + file_path);
                    return false;
                }
                file >> config_json_;
                Logger::getInstance().log(Logger::LogLevel::INFO, "Configuration loaded from JSON file: " + file_path);
                return true;
            } catch (const nlohmann::json::parse_error& e) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, std::string("Failed to parse JSON config file: ") + e.what());
                return false;
            }
        }

        bool Config::loadFromFile(const std::string& file_path) {
            // 判断文件扩展名以确定格式
            size_t dot_pos = file_path.find_last_of(".");
            if(dot_pos == std::string::npos) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Config file has no extension: " + file_path);
                return false;
            }
            std::string extension = file_path.substr(dot_pos + 1);
            // 转换为小写
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

            if(extension == "yaml" || extension == "yml") {
                format = ConfigFormat::YAML;
                return loadYAML(file_path);
            }
            else if(extension == "json") {
                format = ConfigFormat::JSON;
                return loadJSON(file_path);
            }
            else {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Unsupported config file format: " + extension);
                return false;
            }
        }

    } // namespace core
} // namespace rc_vision