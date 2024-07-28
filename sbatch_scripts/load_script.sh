#!/bin/bash
# load_script.sh: 生成计算负载的脚本

# 计算1到10000之间所有数字的平方根
for i in {1..10000}; do
    echo "scale=4; sqrt($i)" | bc > /dev/null
done

# 完成计算后休眠20秒
sleep 20

