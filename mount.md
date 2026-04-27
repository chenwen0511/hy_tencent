# BW1000 节点挂盘方案与实操记录

本文记录本机实际完成的挂盘方案，适用于当前节点：

- 系统盘：`/dev/vda`（`/dev/vda2` 挂载到 `/`）
- 云硬盘：`/dev/vdb`（1T）
- 本地 NVMe：`/dev/nvme0n1`、`/dev/nvme1n1`（各约 6.4TB）

## 1. 盘符识别

```bash
lsblk -o NAME,SIZE,FSTYPE,TYPE,MOUNTPOINT,MODEL
blkid
df -hT
findmnt -D
nvme list
```

识别结论：

- `vda` 为系统盘（已挂载）
- `vdb`、`nvme0n1`、`nvme1n1` 初始均未挂载

## 2. 挂盘方案（已执行）

- `vdb` -> `/data0`（持久数据盘）
- `nvme0n1` -> `/local_nvme0`（本地高速盘）
- `nvme1n1` -> `/local_nvme1`（本地高速盘）

## 3. 实际执行命令

> 注意：以下 `mkfs.ext4 -F` 会格式化目标盘，本机已确认可清空后执行。

```bash
# vdb -> /data0
mkfs.ext4 -F /dev/vdb
mkdir -p /data0
mount /dev/vdb /data0

# nvme0n1 -> /local_nvme0
mkfs.ext4 -F /dev/nvme0n1
mkdir -p /local_nvme0
mount /dev/nvme0n1 /local_nvme0

# nvme1n1 -> /local_nvme1
mkfs.ext4 -F /dev/nvme1n1
mkdir -p /local_nvme1
mount /dev/nvme1n1 /local_nvme1
```

## 4. UUID 与 fstab

实际 UUID：

- `/dev/vdb`：`d0cf0fca-de3f-43de-8bdc-5f4195dc96d4`
- `/dev/nvme0n1`：`0d1802bc-cfb5-417b-9918-954f7f2b6b6d`
- `/dev/nvme1n1`：`2586117e-778f-489f-953a-d641c01d4cb9`

应写入 `/etc/fstab`（请使用真实 UUID，不要使用占位符）：

```bash
UUID=d0cf0fca-de3f-43de-8bdc-5f4195dc96d4 /data0       ext4 defaults,noatime,nofail 0 2
UUID=0d1802bc-cfb5-417b-9918-954f7f2b6b6d /local_nvme0 ext4 defaults,noatime,nofail 0 2
UUID=2586117e-778f-489f-953a-d641c01d4cb9 /local_nvme1 ext4 defaults,noatime,nofail 0 2
```

应用并校验：

```bash
systemctl daemon-reload
mount -a
df -hT | egrep 'data0|local_nvme|Filesystem'
```

## 5. 挂载结果（实测）

```text
/dev/vdb      -> /data0       ext4 1007G
/dev/nvme0n1  -> /local_nvme0 ext4 5.8T
/dev/nvme1n1  -> /local_nvme1 ext4 5.8T
```

结果：三块数据盘均已挂载成功。

## 6. 目录规划（已创建）

```bash
mkdir -p /data0/models /data0/checkpoints /data0/datasets
mkdir -p /local_nvme0/cache /local_nvme0/runs
mkdir -p /local_nvme1/cache /local_nvme1/runs
```

建议用途：

- `/data0/models`：模型权重（持久）
- `/data0/datasets`：训练样本主数据（持久）
- `/data0/checkpoints`：训练产物与断点（持久）
- `/local_nvme*/cache`：缓存与临时中间文件（高性能）
