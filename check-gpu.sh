#!/bin/bash
# 快速查询 p5 集群 GPU 和磁盘状态（并行版）
# Usage: bash check-gpu.sh [host1 host2 ...]
# 默认扫描 p5-2 ~ p5-11（跳过 p5-3）

HOSTS="${@:-p5-1 p5-2 p5-3 p5-4 p5-5 p5-6 p5-7 p5-8 p5-9 p5-10 p5-11}"
TIMEOUT=5

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# 创建临时目录存放各节点结果
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# 并行 SSH 查询所有节点
for host in $HOSTS; do
  (
    result=$(ssh -o ConnectTimeout=$TIMEOUT -o StrictHostKeyChecking=no -o BatchMode=yes -o IdentitiesOnly=yes "$host" '
      # region & instance type (IMDSv2)
      TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null)
      REGION=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
        http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null)
      ITYPE=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
        http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null)

      # GPU: max utilization and total used/total memory across all GPUs
      GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | sort -rn | head -1)
      GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk "{s+=\$1} END {printf \"%.0f\", s/1024}")
      GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
        | awk "{s+=\$1} END {printf \"%.0f\", s/1024}")
      GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

      # NVMe disk
      NVME_FREE=$(df -h /opt/dlami/nvme 2>/dev/null | awk "NR==2 {print \$4}")

      # CPU load (1min)
      LOAD=$(uptime | awk -F"load average: " "{split(\$2,a,\",\"); printf \"%.1f\", a[1]}")

      echo "${REGION}|${ITYPE}|${GPU_UTIL}|${GPU_USED}/${GPU_TOTAL}G|${GPU_COUNT}|${NVME_FREE}|${LOAD}"
    ' 2>/dev/null)

    echo "$result" > "$TMPDIR/$host"
  ) &
done

# 等待所有后台 SSH 完成
wait

# 打印表头
printf "\n${BOLD}%-8s %-16s %-14s %-10s %-8s %-10s %-10s %s${NC}\n" \
  "Host" "Region" "Instance" "GPU Util" "GPU Mem" "NVMe Free" "CPU Load" "Status"
printf '%.0s─' {1..100}; echo

# 按原始顺序输出结果
for host in $HOSTS; do
  result=$(cat "$TMPDIR/$host" 2>/dev/null)

  if [ -z "$result" ]; then
    printf "${RED}%-8s %-16s %-14s %-10s %-8s %-10s %-10s %s${NC}\n" \
      "$host" "-" "-" "-" "-" "-" "-" "❌ OFFLINE"
    continue
  fi

  IFS='|' read -r region itype gpu_util gpu_mem gpu_count nvme_free cpu_load <<< "$result"

  # 判断状态
  gpu_util_num=${gpu_util:-0}
  if [ "$gpu_util_num" -gt 50 ] 2>/dev/null; then
    status="${RED}🔴 BUSY${NC}"
    color=$RED
  elif [ "$gpu_util_num" -gt 10 ] 2>/dev/null; then
    status="${YELLOW}🟡 PARTIAL${NC}"
    color=$YELLOW
  else
    status="${GREEN}🟢 FREE${NC}"
    color=$GREEN
  fi

  # 缩短 region 显示
  case "$region" in
    us-east-2)    region_short="us-east-2/Ohio" ;;
    us-west-2)    region_short="us-west-2/Ore" ;;
    ap-south-1)   region_short="ap-south-1/Mum" ;;
    *)            region_short="$region" ;;
  esac

  printf "${color}%-8s${NC} %-16s %-14s %-10s %-8s %-10s %-10s %b\n" \
    "$host" "$region_short" "$itype" "${gpu_util}%" "$gpu_mem" "$nvme_free" "$cpu_load" "$status"
done

echo ""
