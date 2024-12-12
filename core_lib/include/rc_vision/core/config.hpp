#ifndef RC_VISION_CORE_CONFIG_HPP
#define RC_VISION_CORE_CONFIG_HPP

#include <string>
#include <unordered_map>
#include <yaml-cpp/yaml.h>
#include <nlohmann/json.hpp>
#include "rc_vision/core/logger.hpp"

namespace rc_vision {
    namespace core {

        /**
         * @brief 配置文件支持的格式。
         */
        enum class ConfigFormat {
            YAML, /**< YAML 格式 */
            JSON /**< JSON 格式 */
        };

        /**
         * @brief 配置管理类，支持 YAML 和 JSON 格式的配置文件加载与访问。
         *
         * 该类提供了从文件中加载配置，并通过键值对的方式访问配置参数的功能。默认支持 YAML 格式，也可以通过设置 `format` 为 JSON 来加载 JSON 格式的配置文件。
         */
        class Config {
        public:
            /**
             * @brief 构造函数，默认配置格式为 YAML。
             */
            ConfigFormat format = ConfigFormat::YAML; /**< 配置文件的格式，默认是 YAML */

            /**
             * @brief 从指定文件加载配置。
             *
             * 根据 `format` 成员变量的值，选择相应的加载方法（YAML 或 JSON）。
             *
             * @param file_path 配置文件的路径。
             * @return 成功加载返回 `true`，否则返回 `false`。
             */
            bool loadFromFile(const std::string& file_path);

            /**
             * @brief 根据键获取配置值。
             *
             * 支持模板类型，根据配置文件的格式（YAML 或 JSON）返回对应类型的值。
             *
             * @tparam T 配置值的类型。
             * @param key 配置项的键。
             * @return 配置项的值。
             * @throws std::runtime_error 如果键不存在或格式不支持。
             */
            template<typename T>
            T get(const std::string& key) const;

        private:
            YAML::Node config_yaml_; /**< 存储 YAML 格式的配置数据 */
            nlohmann::json config_json_; /**< 存储 JSON 格式的配置数据 */

            /**
             * @brief 从 YAML 文件加载配置。
             *
             * @param file_path YAML 配置文件的路径。
             * @return 成功加载返回 `true`，否则返回 `false`。
             */
            bool loadYAML(const std::string& file_path);

            /**
             * @brief 从 JSON 文件加载配置。
             *
             * @param file_path JSON 配置文件的路径。
             * @return 成功加载返回 `true`，否则返回 `false`。
             */
            bool loadJSON(const std::string& file_path);
        };

        // 模板方法需要在头文件中定义

        /**
         * @brief 根据键获取配置值的模板实现。
         *
         * 根据 `format` 的值，选择从 YAML 或 JSON 配置中获取值。如果键不存在，将记录错误日志并抛出异常。
         *
         * @tparam T 配置值的类型。
         * @param key 配置项的键。
         * @return 配置项的值。
         * @throws std::runtime_error 如果键不存在或格式不支持。
         */
        template<typename T>
        T Config::get(const std::string& key) const {
            if(format == ConfigFormat::YAML) {
                if(!config_yaml_[key]) {
                    Logger::getInstance().log(Logger::LogLevel::ERROR, "Key not found in YAML config: " + key);
                    throw std::runtime_error("Key not found in YAML config: " + key);
                }
                return config_yaml_[key].as<T>();
            }
            else if(format == ConfigFormat::JSON) {
                if(config_json_.find(key) == config_json_.end()) {
                    Logger::getInstance().log(Logger::LogLevel::ERROR, "Key not found in JSON config: " + key);
                    throw std::runtime_error("Key not found in JSON config: " + key);
                }
                return config_json_.at(key).get<T>();
            }
            else {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Unsupported config format.");
                throw std::runtime_error("Unsupported config format.");
            }
        }

    } // namespace core
} // namespace rc_vision

#endif // RC_VISION_CORE_CONFIG_HPP
