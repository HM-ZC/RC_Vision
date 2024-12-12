# RC Vision 安装指南

欢迎使用 **RC Vision** 机器人视觉库！本指南将帮助您在不同操作系统上安装和配置所需的依赖项，并构建 RC Vision 项目。请按照以下步骤进行操作。

## 目录

1. [简介](#简介)
2. [系统要求](#系统要求)
3. [安装依赖项](#安装依赖项)
   - [Ubuntu](#ubuntu)
   - [macOS](#macos)
   - [Windows](#windows)
4. [构建 RC Vision](#构建-rc-vision)
5. [运行单元测试](#运行单元测试)
6. [rc_vision_core库的API文档](#rc_vision_core库的API文档)
6. [常见问题](#常见问题)
7. [附录](#附录)

## 简介

**RC Vision** 是一个开源的机器人视觉库，支持 ROS1 和 ROS2，提供丰富的视觉算法和工具。本文档将指导您完成这些依赖项的安装以及 RC Vision 的构建和配置。

## 系统要求

- **操作系统**：Ubuntu 20.04 LTS 或更高版本、macOS 10.15 或更高版本、Windows 10 或更高版本
- **编译器**：支持 C++17 的编译器（如 GCC 7.5+、Clang 6+、MSVC 2019+）
- **构建工具**：CMake 3.10 或更高版本
- **其他工具**：Git

## 安装依赖项

### Ubuntu

1. **更新包列表**

    ```bash
    sudo apt-get update
    ```

2. **安装基本开发工具**

    ```bash
    sudo apt-get install -y build-essential cmake git
    ```

3. **安装 OpenCV**

    ```bash
    sudo apt-get install -y libopencv-dev
    ```

4. **安装 Eigen3**

    ```bash
    sudo apt-get install -y libeigen3-dev
    ```

5. **安装 Google Test（GTest）**

    ```bash
    sudo apt-get install -y libgtest-dev
    sudo apt-get install -y cmake
    cd /usr/src/gtest
    sudo cmake .
    sudo make
    sudo mv libg* /usr/lib/
    ```

6. **安装 yaml-cpp**

    ```bash
    sudo apt-get install -y libyaml-cpp-dev
    ```

7. **安装 nlohmann/json**

    ```bash
    sudo apt-get install -y nlohmann-json3-dev
    ```

8. **安装 Graphviz（用于 Doxygen 图表生成）**

    ```bash
    sudo apt-get install -y graphviz
    ```

9. **安装 Python 依赖项**

    ```bash
    sudo apt-get install -y python3-pip
    pip3 install numpy opencv-python
    ```

### macOS

1. **安装 Homebrew**（如果尚未安装）

    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

2. **安装基本开发工具**

    ```bash
    brew install cmake git
    ```

3. **安装 OpenCV**

    ```bash
    brew install opencv
    ```

4. **安装 Eigen3**

    ```bash
    brew install eigen
    ```

5. **安装 Google Test（GTest）**

    ```bash
    brew install googletest
    ```

6. **安装 yaml-cpp**

    ```bash
    brew install yaml-cpp
    ```

7. **安装 nlohmann/json**

    ```bash
    brew install nlohmann-json
    ```

8. **安装 Graphviz**

    ```bash
    brew install graphviz
    ```

9. **安装 Python 依赖项**

    ```bash
    pip3 install numpy opencv-python
    ```

### Windows

1. **安装 Chocolatey**（包管理器，建议使用）

   打开 PowerShell 以管理员身份运行，并执行：

    ```powershell
    Set-ExecutionPolicy Bypass -Scope Process -Force; `
    [System.Net.ServicePointManager]::SecurityProtocol = `
    [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; `
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    ```

2. **安装基本开发工具**

    ```powershell
    choco install -y cmake git visualstudio2019community
    ```

   > **注意**：安装 Visual Studio 时，请选择 "使用 C++ 的桌面开发" 工作负载。

3. **安装 vcpkg（用于安装 C++ 库）**

    ```powershell
    git clone https://github.com/microsoft/vcpkg.git
    cd vcpkg
    .\bootstrap-vcpkg.bat
    ```

4. **安装 C++ 依赖项**

    ```powershell
    .\vcpkg.exe install opencv eigen3 gtest yaml-cpp nlohmann-json
    ```

5. **安装 Python 依赖项**

   下载并安装 [Python](https://www.python.org/downloads/windows/)，然后：

    ```powershell
    python -m pip install --upgrade pip
    pip install numpy opencv-python
    ```

6. **安装 Graphviz**

    ```powershell
    choco install -y graphviz
    ```

## 构建 RC Vision

### 克隆仓库

首先，克隆 RC Vision 仓库：

```bash
cd ~/workspace
git clone https://github.com/HM_ZC/RC_Vision.git
cd RC_Vision
```

### 运行 build.sh 脚本

在项目根目录下，运行以下命令以开始构建过程：

```bash
./build.sh
```

## 运行单元测试

### 运行测试

```bash
ctest --output-on-failure
```

在 Windows 上使用：

```powershell
ctest --output-on-failure
```

## [rc_vision_core库的API文档](docs/html/index.html "API文档")

docs/html/index.html

## 常见问题

## 附录

### 参考资料