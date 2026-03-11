#!/bin/bash
# GIN allreduce test: mew3 (rank 0) + mew2 (rank 1)

export MASTER_ADDR=mew3
export MASTER_PORT=29600
export WORLD_SIZE=2
SELF=$(realpath "$0")

# Worker: set up environment and run the test
if [[ -n "$RANK" ]]; then
  export PATH="$HOME/.local/bin:$PATH"
  export NVCC=/usr/local/cuda-13.1/bin/nvcc
  export NCCL_GIN_ENABLE=1
  export NCCL_SOCKET_IFNAME=enp75s0f1np1
  export NCCL_IB_HCA=mlx5_1
  export NCCL_IB_GID_INDEX=3
  cd /mnt/nfs/edwardhu/cuco
  exec uv run python -m pytest tests/test_gin_allreduce.py::test_gin_allreduce -s
fi

# Coordinator: launch rank 0 locally and rank 1 on mew2
echo "=== GIN allreduce: mew3 (rank 0) + mew2 (rank 1) ==="

RANK=0 timeout 60 bash "$SELF" 2>&1 | sed 's/^/[mew3] /' &
PID0=$!
ssh mew2 "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE RANK=1 timeout 60 bash $SELF" 2>&1 | sed 's/^/[mew2] /' &
PID1=$!

wait $PID0; RC0=$?
wait $PID1; RC1=$?

[ $RC0 -eq 0 ] && [ $RC1 -eq 0 ] && echo "=== PASSED ===" || echo "=== FAILED ==="
exit $((RC0 + RC1))
