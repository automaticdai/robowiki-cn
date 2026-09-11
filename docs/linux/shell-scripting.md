# Shell 脚本与机器人自动化

!!! note "引言"
    机器人系统往往部署在无人值守的环境中，需要在上电后自动完成环境初始化、传感器检测、节点启动等一系列操作。Shell 脚本（Shell Script）能够将这些繁琐的手动步骤固化为可重复执行的自动化流程，显著降低人为操作失误的风险。本文介绍 Bash 核心语法、机器人启动脚本模式、systemd 服务化、udev 设备规则以及常用的实用脚本示例，帮助开发者构建健壮的机器人自动化运维体系。


## Bash 核心语法速查

### 变量、引号与命令替换

Bash 变量无需声明类型，直接赋值即可使用。引用变量时建议用双引号包裹，以防止空格或特殊字符导致解析错误。命令替换（Command Substitution）使用 `$(...)` 语法将命令输出赋值给变量。

```bash
ROBOT_IP="192.168.1.100"
LOG_DIR="/var/log/robot"
DATE=$(date +%Y%m%d_%H%M%S)
echo "日志目录: ${LOG_DIR}/${DATE}"
```

常用变量技巧：

```bash
# 带默认值的变量展开：若 ROS_DISTRO 未设置则使用 noetic
DISTRO="${ROS_DISTRO:-noetic}"

# 字符串截取
DEVICE="/dev/ttyUSB0"
DEVNAME="${DEVICE##*/}"   # 结果：ttyUSB0（去掉最长前缀 /dev/）

# 数组
SENSORS=("lidar" "imu" "camera")
echo "第一个传感器: ${SENSORS[0]}"
echo "传感器数量: ${#SENSORS[@]}"
```

### 条件判断

`[ ]`（`test` 命令）用于条件测试，常见测试选项包括文件存在性、目录、进程等。`[[ ]]` 是 Bash 扩展语法，支持正则匹配，推荐在 Bash 脚本中优先使用。

```bash
if [ -f "/dev/ttyUSB0" ]; then
    echo "串口设备已连接"
elif [ -d "$LOG_DIR" ]; then
    echo "日志目录存在"
fi
```

常用文件测试运算符：

| 运算符 | 含义 |
|--------|------|
| `-f`   | 普通文件存在 |
| `-d`   | 目录存在 |
| `-e`   | 文件或目录存在 |
| `-r`   | 可读 |
| `-w`   | 可写 |
| `-x`   | 可执行 |
| `-s`   | 文件非空 |
| `-z`   | 字符串为空 |
| `-n`   | 字符串非空 |

进程存在性检测示例：

```bash
# 检查进程是否运行（按进程名）
if pgrep -x "rosmaster" &>/dev/null; then
    echo "rosmaster 正在运行"
else
    echo "rosmaster 未启动"
fi

# 检查端口是否监听
if ss -tlnp | grep -q ":11311"; then
    echo "ROS Master 端口 11311 已监听"
fi
```

### 循环

**while 循环**常用于等待某个条件满足，例如等待 ROS Master（机器人操作系统主节点）启动：

```bash
# 等待 ROS Master 启动
while ! rostopic list &>/dev/null; do
    echo "等待 ROS Master..."
    sleep 1
done
echo "ROS Master 已就绪"
```

带超时的等待循环：

```bash
TIMEOUT=30
COUNT=0
while ! rostopic list &>/dev/null; do
    if [ "$COUNT" -ge "$TIMEOUT" ]; then
        echo "错误：等待 ROS Master 超时（${TIMEOUT}秒）" >&2
        exit 1
    fi
    echo "等待 ROS Master... (${COUNT}/${TIMEOUT})"
    sleep 1
    COUNT=$((COUNT + 1))
done
```

**for 循环**遍历列表：

```bash
# 遍历传感器话题并检查发布频率
TOPICS=("/scan" "/imu/data" "/camera/image_raw")
for topic in "${TOPICS[@]}"; do
    hz=$(rostopic hz "$topic" --window=10 2>/dev/null | grep "average rate" | awk '{print $3}')
    echo "话题 $topic 频率: ${hz:-未知} Hz"
done
```

**until 循环**（条件为假时继续执行，与 while 相反）：

```bash
# 等待设备文件出现
until [ -e "/dev/lidar" ]; do
    echo "等待激光雷达设备..."
    sleep 2
done
echo "激光雷达设备已就绪"
```

### 函数定义

函数将重复逻辑封装为可复用单元，`local` 关键字声明局部变量，避免污染全局命名空间。函数通过返回值（`return`，范围 0-255）或输出（`echo`）传递结果。

```bash
check_dependency() {
    local pkg=$1
    if ! command -v "$pkg" &>/dev/null; then
        echo "错误：$pkg 未安装" >&2
        exit 1
    fi
}
check_dependency ros
check_dependency python3
```

带返回值的函数：

```bash
# 返回 0 表示成功，1 表示失败
is_ros_running() {
    if rostopic list &>/dev/null; then
        return 0
    else
        return 1
    fi
}

if is_ros_running; then
    echo "ROS 运行正常"
fi
```

### 错误处理与日志

生产级脚本应启用严格模式并统一日志格式：

```bash
#!/bin/bash
set -euo pipefail
# -e: 遇错即退（Exit on error）
# -u: 使用未定义变量时报错（Undefined variable error）
# -o pipefail: 管道中任意命令失败则整体失败

# 日志函数
LOG_FILE="/var/log/robot/startup.log"

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO]  $*" | tee -a "$LOG_FILE"
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN]  $*" | tee -a "$LOG_FILE" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" | tee -a "$LOG_FILE" >&2
}

# 捕获退出信号，执行清理操作
cleanup() {
    log_info "脚本退出，执行清理..."
    kill $(jobs -p) 2>/dev/null || true
}
trap cleanup EXIT INT TERM
```


## 本章内容导览

Shell 脚本章节按「语法基础 → 启动流程 → 运维自动化 → 现场排查 → 语法速查」的顺序组织：

| 页面 | 主要内容 |
|------|---------|
| [Shell 脚本与机器人自动化](shell-scripting.md) | Bash 核心语法速查、机器人场景下的常用写法与健壮性约定 |
| [机器人启动脚本模式](shell-scripting-robot-startup.md) | ROS 1/ROS 2 启动脚本、起飞前检查、安全停止 |
| [自动化模式](shell-scripting-automation-patterns.md) | 开机自启、systemd 服务、看门狗、日志轮转、备份、CI/CD |
| [运维工具箱](shell-scripting-ops-toolkit.md) | 资源监控、进程管理、网络调试、USB 与串口设备、Docker |
| [Bash 语法完整参考](shell-scripting-full-reference.md) | 变量、条件、循环、函数、字符串、正则、ShellCheck 与风格指南 |
| [Linux 常用命令](commands.md) | 命令行基础 |


## 参考资料

- GNU Bash 官方手册：https://www.gnu.org/software/bash/manual/bash.html
- Advanced Bash-Scripting Guide（高级 Bash 脚本指南）：https://tldp.org/LDP/abs/html/
- systemd 服务单元文档：https://www.freedesktop.org/software/systemd/man/systemd.service.html
- udev 规则编写指南：https://www.reactivated.net/writing_udev_rules.html
- ROS Wiki - roslaunch：http://wiki.ros.org/roslaunch
- ROS 2 文档 - 启动系统：https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Launch-Main.html
- rosbag 使用手册：http://wiki.ros.org/rosbag/Commandline
