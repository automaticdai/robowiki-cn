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


## 机器人启动脚本模式

### ROS 1 自动启动脚本

以下是一个完整的 ROS 1（机器人操作系统第一版）自动启动脚本，适用于搭载 Ubuntu 20.04 + ROS Noetic 的机器人平台：

```bash
#!/bin/bash
# robot_start.sh - 机器人系统启动脚本
# 用法: ./robot_start.sh [--debug]
set -e  # 遇错即退

# ---- 配置区 ----
ROS_DISTRO_NAME="noetic"
CATKIN_WS="$HOME/catkin_ws"
LOG_DIR="/var/log/robot"
ROBOT_IP="192.168.1.100"

# ---- 初始化日志 ----
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/startup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "[$(date)] 机器人系统启动脚本开始执行"

# ---- 加载 ROS 环境 ----
source /opt/ros/${ROS_DISTRO_NAME}/setup.bash
source ${CATKIN_WS}/devel/setup.bash
echo "ROS 环境已加载：$ROS_DISTRO"

# ---- 启动 roscore（后台运行）----
roscore &
ROSCORE_PID=$!
echo "roscore PID: $ROSCORE_PID"

# ---- 等待 Master 就绪 ----
echo "等待 ROS Master 启动..."
sleep 3
TIMEOUT=15
COUNT=0
while ! rostopic list &>/dev/null; do
    COUNT=$((COUNT + 1))
    if [ "$COUNT" -ge "$TIMEOUT" ]; then
        echo "错误：ROS Master 启动超时" >&2
        exit 1
    fi
    sleep 1
done
echo "ROS Master 已就绪"

# ---- 启动传感器节点 ----
roslaunch robot_bringup sensors.launch &
SENSORS_PID=$!
echo "传感器节点 PID: $SENSORS_PID"
sleep 2

# ---- 启动导航栈（Navigation Stack）----
roslaunch navigation_stack navigation.launch &
NAV_PID=$!
echo "导航栈 PID: $NAV_PID"

echo "[$(date)] 机器人系统启动完成"

# 等待所有后台进程
wait
```

### ROS 2 自动启动脚本

ROS 2（机器人操作系统第二版）取消了 roscore 的概念，通过 DDS（数据分发服务）实现去中心化通信：

```bash
#!/bin/bash
# robot_start_ros2.sh - ROS 2 机器人启动脚本
set -euo pipefail

ROS2_DISTRO="humble"
ROS2_WS="$HOME/ros2_ws"
LOG_DIR="/var/log/robot"

mkdir -p "$LOG_DIR"

# 加载 ROS 2 环境
source /opt/ros/${ROS2_DISTRO}/setup.bash
source ${ROS2_WS}/install/setup.bash

echo "ROS 2 发行版: $ROS_DISTRO"

# 设置 DDS（数据分发服务）域 ID，避免多机器人相互干扰
export ROS_DOMAIN_ID=42

# 启动机器人主 launch 文件
ros2 launch robot_bringup robot.launch.py \
    use_sim_time:=false \
    robot_name:=my_robot &

LAUNCH_PID=$!
echo "Launch 进程 PID: $LAUNCH_PID"

# 等待关键话题出现
echo "等待传感器话题..."
TIMEOUT=30
COUNT=0
while ! ros2 topic list 2>/dev/null | grep -q "/scan"; do
    COUNT=$((COUNT + 1))
    if [ "$COUNT" -ge "$TIMEOUT" ]; then
        echo "警告：激光雷达话题未出现，继续启动..." >&2
        break
    fi
    sleep 1
done

echo "ROS 2 系统启动完成"
wait "$LAUNCH_PID"
```

### 环境检测与依赖校验

启动前的环境检测脚本，可在正式启动脚本的开头调用：

```bash
#!/bin/bash
# preflight_check.sh - 机器人起飞前检查（Preflight Check）

# 检查 ROS 环境是否已加载
check_ros_env() {
    if [ -z "$ROS_DISTRO" ]; then
        echo "错误：ROS 环境未加载，请先 source setup.bash" >&2
        exit 1
    fi
    echo "ROS 发行版: $ROS_DISTRO"
}

# 检查串口设备（Serial Port）是否存在
check_serial_device() {
    local device=$1
    if [ ! -e "$device" ]; then
        echo "错误：串口设备 $device 不存在" >&2
        echo "已连接的串口设备："
        ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "无设备"
        exit 1
    fi
    echo "串口设备 $device 已就绪"
}

# 检查磁盘空间（单位：MB）
check_disk_space() {
    local path=$1
    local required_mb=$2
    local available_mb
    available_mb=$(df -m "$path" | awk 'NR==2 {print $4}')
    if [ "$available_mb" -lt "$required_mb" ]; then
        echo "错误：$path 可用空间不足（需要 ${required_mb}MB，当前 ${available_mb}MB）" >&2
        exit 1
    fi
    echo "磁盘空间检查通过：$path 可用 ${available_mb}MB"
}

# 检查网络连通性
check_network() {
    local host=$1
    if ! ping -c 1 -W 2 "$host" &>/dev/null; then
        echo "警告：无法连接到 $host" >&2
        return 1
    fi
    echo "网络连通性检查通过：$host 可达"
}

# 执行所有检查
check_ros_env
check_serial_device "/dev/ttyUSB0"
check_disk_space "/data" 1024     # 至少 1GB 空余
check_network "192.168.1.1"

echo "所有预检通过，系统可以启动"
```


## systemd 服务化机器人程序

将机器人启动脚本注册为 systemd 服务，可实现开机自启、崩溃自重启、日志持久化等功能。systemd 是现代 Linux 发行版的初始化系统（Init System）。

### 创建服务单元文件

创建 `/etc/systemd/system/robot.service`：

```ini
[Unit]
Description=Robot Main Service
Documentation=https://your-robot-wiki.example.com
After=network.target
Wants=network-online.target

[Service]
Type=forking
User=robot
Group=robot
WorkingDirectory=/home/robot

# 环境变量文件（每行一个 KEY=VALUE）
EnvironmentFile=-/etc/robot/environment

# 启动和停止命令
ExecStart=/home/robot/scripts/robot_start.sh
ExecStop=/home/robot/scripts/robot_stop.sh

# 崩溃后自动重启策略
Restart=on-failure
RestartSec=5

# 启动超时限制
TimeoutStartSec=60
TimeoutStopSec=30

# 日志输出到 systemd journal
StandardOutput=journal
StandardError=journal
SyslogIdentifier=robot

[Install]
WantedBy=multi-user.target
```

对应的停止脚本 `/home/robot/scripts/robot_stop.sh`：

```bash
#!/bin/bash
# robot_stop.sh - 安全停止机器人系统
echo "正在停止机器人系统..."

# ROS 1：发送关闭信号
if command -v rosnode &>/dev/null; then
    rosnode kill --all 2>/dev/null || true
    sleep 2
fi

# 终止所有相关进程
pkill -f "roslaunch" 2>/dev/null || true
pkill -f "roscore" 2>/dev/null || true
pkill -f "ros2 launch" 2>/dev/null || true

echo "机器人系统已停止"
```

### 服务管理命令

```bash
# 重新加载 systemd 配置（修改 .service 文件后必须执行）
sudo systemctl daemon-reload

# 开机自启（启用服务）
sudo systemctl enable robot.service

# 立即启动服务
sudo systemctl start robot.service

# 停止服务
sudo systemctl stop robot.service

# 重启服务
sudo systemctl restart robot.service

# 查看服务运行状态
sudo systemctl status robot.service

# 查看实时日志（-f 表示跟随最新输出）
journalctl -u robot.service -f

# 查看最近 100 行日志
journalctl -u robot.service -n 100

# 查看本次启动的日志
journalctl -u robot.service -b
```

### 多服务依赖编排

当机器人系统由多个独立服务组成时，可通过 `After`、`Requires`、`Wants` 字段声明依赖关系：

```ini
# /etc/systemd/system/robot-nav.service
[Unit]
Description=Robot Navigation Service
After=robot-sensors.service
Requires=robot-sensors.service

[Service]
Type=simple
User=robot
ExecStart=/home/robot/scripts/start_navigation.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
```


## udev 规则固定串口名

Linux 系统中，USB 转串口设备（如 CP2102、FT232、CH340）的设备节点（/dev/ttyUSBx）编号在每次插拔后可能发生变化（即"设备名漂移"问题）。通过 udev 规则（udev Rules）可为特定设备绑定固定的符号链接（Symbolic Link）名称。

### 查找设备属性

在设备接入后，使用以下命令获取设备的厂商 ID（Vendor ID）、产品 ID（Product ID）及序列号：

```bash
# 查看 /dev/ttyUSB0 的完整属性树
udevadm info --name=/dev/ttyUSB0 --attribute-walk | grep -E "idVendor|idProduct|serial"

# 简洁方式：直接查看设备属性
udevadm info -q property /dev/ttyUSB0

# 查看所有已连接的 USB 设备及其 ID
lsusb
```

示例输出（以禾顿 FT232 为例）：

```
    ATTRS{idVendor}=="0403"
    ATTRS{idProduct}=="6001"
    ATTRS{serial}=="A9M8JKLP"
```

### 创建 udev 规则文件

创建 `/etc/udev/rules.d/99-robot.rules`：

```
# 激光雷达（Light Detection and Ranging，Hokuyo UTM-30LX）
SUBSYSTEM=="tty", ATTRS{idVendor}=="0f0d", ATTRS{idProduct}=="0059", SYMLINK+="lidar"

# IMU（惯性测量单元，Inertial Measurement Unit，LPMS-B2）
SUBSYSTEM=="tty", ATTRS{idVendor}=="1dcf", ATTRS{idProduct}=="0002", SYMLINK+="imu"

# 底盘串口（FT232 芯片）
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", SYMLINK+="robot_base"

# 通过序列号区分同型号设备（同一 VID/PID 但序列号不同）
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", ATTRS{serial}=="A9M8JKLP", SYMLINK+="robot_arm"
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", ATTRS{serial}=="B2N7XQRT", SYMLINK+="robot_gripper"
```

规则说明：设备插入后将在 `/dev/lidar`、`/dev/imu`、`/dev/robot_base` 创建指向实际设备节点的软链接，脚本中统一使用固定名称即可。

### 重新加载规则

```bash
# 重新加载 udev 规则并触发设备事件
sudo udevadm control --reload && sudo udevadm trigger

# 验证符号链接是否创建成功
ls -la /dev/lidar /dev/imu /dev/robot_base

# 在不重新插拔的情况下手动触发单个设备规则
sudo udevadm trigger --name-match=/dev/ttyUSB0
```

### 配置设备权限

默认情况下，串口设备需要 `dialout` 组权限。将用户加入该组后无需 sudo 即可访问串口：

```bash
# 将当前用户加入 dialout 组（需重新登录生效）
sudo usermod -aG dialout $USER

# 或在 udev 规则中直接设置设备权限
# SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ..., MODE="0666", GROUP="dialout"
```


## 实用脚本示例

### 进程守护

确保关键 ROS 节点在崩溃后自动重启（此功能也可由 systemd 的 `Restart=on-failure` 实现，但有时需要在 ROS 层面做更细粒度的控制）：

```bash
#!/bin/bash
# node_watchdog.sh - ROS 节点守护脚本（Watchdog）
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

NODE_NAME="/slam_node"
LAUNCH_CMD="roslaunch slam_pkg slam.launch"
CHECK_INTERVAL=10  # 检查间隔（秒）
RESTART_DELAY=5    # 重启前等待（秒）

echo "守护进程启动，监控节点：$NODE_NAME"

while true; do
    if ! rosnode list 2>/dev/null | grep -q "^${NODE_NAME}$"; then
        echo "$(date '+%Y-%m-%d %H:%M:%S'): $NODE_NAME 未运行，正在重启..."
        $LAUNCH_CMD &
        sleep "$RESTART_DELAY"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S'): $NODE_NAME 运行正常"
    fi
    sleep "$CHECK_INTERVAL"
done
```

### rosbag 定时录制脚本

rosbag 是 ROS 的话题数据录制工具，用于调试和数据采集：

```bash
#!/bin/bash
# rosbag_record.sh - 定时录制传感器数据
DURATION=300  # 录制时长（秒），此处为 5 分钟
TOPICS="/camera/image_raw /scan /odom /tf /imu/data"
OUTPUT_DIR="/data/rosbag/$(date +%Y%m%d)"

mkdir -p "$OUTPUT_DIR"
echo "开始录制，输出目录：$OUTPUT_DIR"
echo "录制话题：$TOPICS"
echo "录制时长：${DURATION} 秒"

rosbag record \
    -O "${OUTPUT_DIR}/record_$(date +%H%M%S).bag" \
    --duration="$DURATION" \
    --split --size=1024 \
    $TOPICS

echo "录制完成：$OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
```

### 机器人网络连通性检查

```bash
#!/bin/bash
# network_check.sh - 检查机器人网络中各设备的连通性
# 格式：IP地址:设备名称
TARGETS=(
    "192.168.1.1:路由器"
    "192.168.1.50:相机"
    "192.168.1.51:LiDAR"
    "192.168.1.100:机械臂控制器"
)

FAIL_COUNT=0

echo "=== 机器人网络连通性检查 $(date) ==="
for entry in "${TARGETS[@]}"; do
    ip="${entry%%:*}"
    name="${entry##*:}"
    if ping -c 1 -W 1 "$ip" &>/dev/null; then
        echo "[在线] $name ($ip)"
    else
        echo "[离线] $name ($ip)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo "=== 检查完成，${FAIL_COUNT} 个设备不可达 ==="
exit "$FAIL_COUNT"
```

### 日志清理脚本

机器人长期运行会积累大量日志文件，定期清理是必要的运维操作：

```bash
#!/bin/bash
# log_cleanup.sh - 清理过期日志
LOG_DIR="/var/log/robot"
ROSBAG_DIR="/data/rosbag"
KEEP_DAYS=7  # 保留最近 7 天的日志

echo "清理 $KEEP_DAYS 天前的日志文件..."

# 清理机器人系统日志
find "$LOG_DIR" -name "*.log" -mtime +"$KEEP_DAYS" -exec rm -v {} \;

# 清理 rosbag 录制文件（超过 30 天）
find "$ROSBAG_DIR" -name "*.bag" -mtime +30 -exec rm -v {} \;

# 清理 ROS 自身日志（~/.ros/log/）
if [ -d "$HOME/.ros/log" ]; then
    find "$HOME/.ros/log" -mindepth 1 -maxdepth 1 -type d -mtime +"$KEEP_DAYS" \
        -exec rm -rf {} \;
    echo "ROS 日志清理完成"
fi

echo "磁盘使用情况："
df -h "$LOG_DIR" "$ROSBAG_DIR" 2>/dev/null
```

将清理脚本加入 cron（定时任务）计划：

```bash
# 编辑当前用户的 crontab
crontab -e

# 每天凌晨 2 点执行清理
0 2 * * * /home/robot/scripts/log_cleanup.sh >> /var/log/robot/cleanup.log 2>&1
```

### 多机器人状态汇总脚本

在多机器人系统中，管理节点需要轮询各机器人的状态：

```bash
#!/bin/bash
# fleet_status.sh - 机器人集群状态汇总
ROBOTS=(
    "robot1:192.168.1.101"
    "robot2:192.168.1.102"
    "robot3:192.168.1.103"
)
SSH_USER="robot"
SSH_OPTS="-o ConnectTimeout=3 -o StrictHostKeyChecking=no"

echo "=== 机器人集群状态 $(date) ==="
printf "%-12s %-16s %-10s %-20s\n" "名称" "IP地址" "网络" "系统状态"
echo "------------------------------------------------------------"

for entry in "${ROBOTS[@]}"; do
    name="${entry%%:*}"
    ip="${entry##*:}"

    if ping -c 1 -W 1 "$ip" &>/dev/null; then
        network="在线"
        status=$(ssh $SSH_OPTS "$SSH_USER@$ip" \
            "systemctl is-active robot.service 2>/dev/null || echo '未知'" 2>/dev/null)
    else
        network="离线"
        status="N/A"
    fi

    printf "%-12s %-16s %-10s %-20s\n" "$name" "$ip" "$network" "$status"
done
```


## 参考资料

- GNU Bash 官方手册：https://www.gnu.org/software/bash/manual/bash.html
- Advanced Bash-Scripting Guide（高级 Bash 脚本指南）：https://tldp.org/LDP/abs/html/
- systemd 服务单元文档：https://www.freedesktop.org/software/systemd/man/systemd.service.html
- udev 规则编写指南：https://www.reactivated.net/writing_udev_rules.html
- ROS Wiki - roslaunch：http://wiki.ros.org/roslaunch
- ROS 2 文档 - 启动系统：https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Launch-Main.html
- rosbag 使用手册：http://wiki.ros.org/rosbag/Commandline
