#include "rc_vision/core/logger.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>

using namespace rc_vision::core;

TEST(LoggerTest, SingletonInstance) {
Logger& logger1 = Logger::getInstance();
Logger& logger2 = Logger::getInstance();
EXPECT_EQ(&logger1, &logger2);
}

TEST(LoggerTest, LogLevelFiltering) {
Logger& logger = Logger::getInstance();
logger.setLogLevel(Logger::LogLevel::WARN);

// Redirect std::cout to a stringstream to capture output
std::streambuf* original_cout = std::cout.rdbuf();
std::ostringstream captured_output;
std::cout.rdbuf(captured_output.rdbuf());

LOG_INFO("This INFO message should not appear");
LOG_WARN("This WARN message should appear");
LOG_ERROR("This ERROR message should appear");

// Restore original std::cout
std::cout.rdbuf(original_cout);

std::string output = captured_output.str();
EXPECT_NE(output.find("WARN"), std::string::npos);
EXPECT_NE(output.find("ERROR"), std::string::npos);
EXPECT_EQ(output.find("INFO"), std::string::npos);
}

TEST(LoggerTest, LogFileOutput) {
Logger& logger = Logger::getInstance();
std::string log_file = "test_log.log";

// Ensure the log file is clean
if (std::filesystem::exists(log_file)) {
std::filesystem::remove(log_file);
}

ASSERT_TRUE(logger.setLogFile(log_file, 1024)); // 1KB for testing

LOG_INFO("Test log entry");

logger.shutdown();

std::ifstream infile(log_file);
ASSERT_TRUE(infile.is_open());
std::string line;
std::getline(infile, line);
infile.close();

EXPECT_NE(line.find("Test log entry"), std::string::npos);

// Clean up
std::filesystem::remove(log_file);
}

TEST(LoggerTest, LogRotation) {
Logger& logger = Logger::getInstance();
std::string log_file = "test_log_rotate.log";

// Ensure the log file is clean
if (std::filesystem::exists(log_file)) {
std::filesystem::remove(log_file);
}

ASSERT_TRUE(logger.setLogFile(log_file, 100)); // 100 bytes for quick rotation

LOG_INFO("First log entry");
LOG_INFO("Second log entry that should trigger rotation");

logger.shutdown();

// Check that rotation occurred
size_t rotated_files = 0;
for(auto& p: std::filesystem::directory_iterator(".")) {
if(p.path().string().find("test_log_rotate.log.") != std::string::npos) {
rotated_files++;
}
}

EXPECT_GT(rotated_files, 0);

// Clean up
std::filesystem::remove(log_file);
for(auto& p: std::filesystem::directory_iterator(".")) {
if(p.path().string().find("test_log_rotate.log.") != std::string::npos) {
std::filesystem::remove(p.path());
}
}
}