# 机器人启动脚本模式

!!! note "引言"
    机器人上电后需要按固定顺序拉起一系列进程：先加载 ROS 环境，再启动 Master 或设置 DDS 域，等待关键话题就绪后才能启动导航与感知节点，任何一步提前都会导致节点连接失败。把这套顺序固化为启动脚本，是从「开发机上手动开几个终端」走向「机器人可独立上电运行」的第一步。与之配套的还有起飞前检查（Preflight Check）脚本与安全停止脚本。本页面给出 ROS 1 与 ROS 2 的启动脚本模板及检查脚本写法。


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


## 参考资料

- GNU Bash 官方手册：https://www.gnu.org/software/bash/manual/bash.html
- ROS Wiki, [roslaunch](http://wiki.ros.org/roslaunch)
- ROS 2 Documentation, [Launch system](https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/)
- [Shell 脚本与机器人自动化](shell-scripting.md)
- [自动化模式](shell-scripting-automation-patterns.md) —— systemd 服务与看门狗
- [运维工具箱](shell-scripting-ops-toolkit.md) —— udev 规则与串口设备管理
