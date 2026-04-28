## 海光 BW1000 常用命令速查

本文汇总自 `BW1000.pdf` 与 `README.md`，用于现场快速排查与训练/推理启动。

### 1) 系统与驱动基础

```bash
# 配置 hyhal 工具链
echo 'export PATH="/opt/hyhal/bin/:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 驱动模块检查（6.3+ 常见 hycu；老版本常见 hydcu）
lsmod | grep -E "hydcu|hycu"

# 卡状态与拓扑
hy-smi
hy-smi --showtopo
```

### 2) 网络/RDMA 与 IP 排查

```bash
# 查看 RDMA 设备与网卡映射
ibdev2netdev

# 查看 bond/业务网卡 IP（server/client 对接前必查）
ifconfig bond0
ifconfig eth0
hostname -I

# 查看 PCIe 拓扑（多 bond 并发压测前建议）
lspci -tv
```

### 3) 多机/RCCL IP 清单（模板）

```bash
cd /workspace/tools
cat > ip_eth0.txt << EOF
10.0.0.10
10.0.0.8
10.0.0.9
10.0.0.1
EOF

# 按网络拓扑排序
bash get_rdma_order_by_ip.sh ip_eth0.txt
cat hostfile.txt
```

### 4) 多机 Ray（LLM 推理常见）

```bash
# 头节点
ray start --head --node-ip-address=<head_ip> --port=1222 --num-gpus=8 --num-cpus=32

# 其他节点
ray start --address='<head_ip>:1222' --num-gpus=8 --num-cpus=32

# 检查集群
ray status
```

### 5) 常见问题定位

```bash
# docker / 驱动服务状态
systemctl status docker

# 容器内外 DCU 可见性
hy-smi
docker exec -it <container_name> bash -lc "hy-smi"

# 训练/推理日志
tail -f *.log
```

