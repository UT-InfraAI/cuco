"""NCCL Device API documentation modules for kernel optimization prompts.

External references (NCCL 2.28.9):
- Device API Reference: https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2289/user-guide/docs/api/device.html#gin
- Device API Usage Guide: https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2289/user-guide/docs/usage/deviceapi.html
"""

NCCL_DEVICE_API_REFERENCE = """## NCCL Device API Overview

Host-side setup is complete. Focus on device-kernel usage only.

**Core Concepts:**
- **LSA (Load/Store Accessible)**: Direct memory access to local GPU peers
- **GIN (GPU-Initiated Networking)**: One-sided network transfers to remote GPUs
- **Teams**: Subsets of ranks - World (all), LSA (local), Rail (network)
- **Thread Groups**: Block (`ncclCoopCta`), Warp (`ncclCoopWarp`), Thread (`ncclCoopThread`)

**Key Principles:**
- Only use device APIs inside kernels (no host-side calls)
- Synchronize appropriately between writes and reads
"""


NCCL_LSA_API_DOC = """## NCCL LSA (Load/Store Accessible) API Documentation

### Overview
The LSA API enables device-side memory synchronization and peer access within local teams of GPUs.
All LSA functionality operates on the device side only.

### Core Components

#### ncclLsaBarrierSession
**Purpose**: Manages memory barrier synchronization across LSA team members.

**Constructors**:
```cpp
// General constructor (any LSA-accessible team)
ncclLsaBarrierSession(Coop coop, ncclDevComm const &comm, ncclTeam team,
                      ncclLsaBarrierHandle handle, uint32_t index,
                      bool multimem = false, ncclMultimemHandle mmHandle = {})

// Convenience constructor (LSA team shorthand)
ncclLsaBarrierSession(Coop coop, ncclDevComm const &comm,
                      ncclTeamTagLsa tag, uint32_t index,
                      bool multimem = false)
```

**Parameters**:
- `coop`: Cooperative group participating in the barrier
- `comm`: Device communicator created via `ncclDevCommCreate()`
- `team`: Team for barrier scope (general constructor)
- `handle`: LSA barrier handle from devComm (e.g., `devComm.lsaBarrier`)
- `tag`: `ncclTeamTagLsa{}` for convenience constructor
- `index`: Barrier identifier (typically `blockIdx.x` for CTA uniqueness)
- `multimem`: Enable NVLink multicast (Hopper+, optional)
- `mmHandle`: Multimem handle (optional, general constructor only)

**Key Methods**:
- `arrive(Coop, cuda::memory_order)`: Signals thread arrival at barrier
- `wait(Coop, cuda::memory_order)`: Blocks until all team members arrive
- `sync(Coop, cuda::memory_order)`: Combined arrive-and-wait operation

### Memory Access Functions

#### ncclGetPeerPointer
Returns load/store accessible pointer to peer device memory within a window.
Returns NULL if peer is unreachable via LSA.

**Signature**:
```cpp
void* ncclGetPeerPointer(ncclWindow_t window, size_t offset, int peerRank)
```

#### ncclGetLsaPointer
Similar to peer pointer but uses LSA rank indexing directly within the local team.

**Signature**:
```cpp
void* ncclGetLsaPointer(ncclWindow_t window, size_t offset, int lsaRank)
```

#### ncclGetLocalPointer
Convenience function returning pointer to current device's memory within the window.

**Signature**:
```cpp
void* ncclGetLocalPointer(ncclWindow_t window, size_t offset)
```

### Usage Patterns

#### Basic LSA Barrier Synchronization
```cpp
// Create barrier session
ncclLsaBarrierSession<ncclCoopCta> lsaBar {
    ncclCoopCta(),
    devComm,
    ncclTeamLsa(devComm),
    devComm.lsaBarrier,
    blockIdx.x  // barrier ID
};

// Perform memory operations...

// Synchronize with relaxed ordering
lsaBar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

// More operations...

// Synchronize with release semantics
lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
```

#### LSA Peer Memory Access
```cpp
ncclTeam lsa = ncclTeamLsa(devComm);

// Get local memory pointer
float* local = (float*)ncclGetLocalPointer(window, offset);

// Access LSA peer memory directly
for (int lp = 0; lp < lsa.nRanks; ++lp) {
    float* peer = (float*)ncclGetLsaPointer(window, offset, lp);
    // Direct load/store to peer memory
    peer[idx] = local[idx];
}
```

### Memory Ordering
LSA barriers support standard C++ memory ordering semantics:
- `cuda::memory_order_relaxed`: No synchronization guarantee
- `cuda::memory_order_acquire`: Previous writes by other threads are visible
- `cuda::memory_order_release`: Current writes are visible to other threads
- `cuda::memory_order_acq_rel`: Combined acquire-release semantics
- `cuda::memory_order_seq_cst`: Sequentially consistent ordering
"""


NCCL_GIN_API_DOC = """## NCCL GIN (GPU-Initiated Networking) API Documentation

### Overview
The GIN API enables device-initiated, one-sided network communication with remote peers.
GIN is supported since NCCL 2.28.7. All operations are device-side only.

### Core Components

#### ncclGin Class
**Purpose**: Manages device-initiated one-sided network transfers to remote peers.

**Constructor**:
```cpp
ncclGin(ncclDevComm const &comm, int contextIndex)
```

**Parameters**:
- `comm`: Device communicator created via `ncclDevCommCreate()`
- `contextIndex`: Network communication channel identifier. Multiple contexts allow spreading traffic across connections to avoid bottlenecks.

### Data Transfer Operations

#### put
Executes a device-initiated, one-sided data transfer from local buffer to remote buffer on a peer.

**Signature**:
```cpp
void put(ncclTeam team, int peer,
         ncclWindow_t dstWnd, size_t dstOffset,
         ncclWindow_t srcWnd, size_t srcOffset,
         size_t bytes,
         RemoteAction remoteAction = ncclGin_None{},
         LocalAction localAction = ncclGin_None{},
         Coop coop = ncclCoopThread{},
         DescriptorSmem descriptor = ncclGin_None{},
         cuda::thread_scope alreadyReleased = cuda::thread_scope_thread,
         cuda::thread_scope expected_scope = cuda::thread_scope_device)
```

**Key Parameters**:
- `team`: Team handle (e.g., from `ncclTeamWorld()`)
- `peer`: Target peer rank within the team
- `dstWnd`, `dstOffset`: Destination window and offset
- `srcWnd`, `srcOffset`: Source window and offset
- `bytes`: Number of bytes to transfer
- `remoteAction`: Optional signal increment on remote peer (e.g., `ncclGin_SignalInc{signalIndex}`)
- `localAction`: Optional counter increment on local peer (e.g., `ncclGin_CounterInc{counterIndex}`)

**Ordering Guarantee**: Visibility of attached signals implies visibility of the put data and all preceding puts to the same peer (when using the same GIN context).

#### flush
Ensures pending transfers are locally consumed, meaning their source buffers are safe to reuse.

**Signature**:
```cpp
void flush(Coop coop, cuda::memory_order ord = cuda::memory_order_acquire)
```

### Signals (Remote Actions)

Signals are 64-bit integer values manipulated atomically, used for remote notification.
The signal index type is a 32-bit identifier; the signal *value* read/written is 64-bit.

#### Signal Types
- `ncclGinSignal_t`: 32-bit signal index (`uint32_t`); the signal value is `uint64_t`
- `ncclGin_SignalInc`: Increment signal by 1 on remote peer
- `ncclGin_SignalAdd`: Add specific value to signal on remote peer

#### Signal Methods

**readSignal**: Read current signal value
```cpp
uint64_t readSignal(ncclGinSignal_t signal, int bits = 64,
                    cuda::memory_order ord = cuda::memory_order_acquire)
```

**waitSignal**: Wait until signal reaches at least the specified value
```cpp
void waitSignal(Coop coop, ncclGinSignal_t signal, uint64_t least,
                int bits = 64, cuda::memory_order ord = cuda::memory_order_acquire)
```

**resetSignal**: Reset signal to zero
```cpp
void resetSignal(ncclGinSignal_t signal)
```

**signal**: Send standalone signal without data transfer
```cpp
void signal(ncclTeam team, int peer, RemoteAction remoteAction,
            Coop coop = ncclCoopThread{},
            DescriptorSmem descriptor = ncclGin_None{},
            cuda::thread_scope alreadyReleased = cuda::thread_scope_thread,
            cuda::thread_scope expected_scope = cuda::thread_scope_device)
```

**Rolling Comparison**: Signal comparisons use rolling logic to handle unsigned overflow.

### Counters (Local Actions)

Counters track local completion of operations, limited to 56-bit values.

#### Counter Types
- `ncclGinCounter_t`: 32-bit counter index (`uint32_t`); counter values are `uint64_t` (max 56 bits usable)
- `ncclGin_CounterInc`: Increment counter locally on source peer

#### Counter Methods

**readCounter**: Read current counter value
```cpp
uint64_t readCounter(ncclGinCounter_t counter, int bits = 56,
                     cuda::memory_order ord = cuda::memory_order_acquire)
```

**waitCounter**: Wait until counter reaches at least the specified value
```cpp
void waitCounter(Coop coop, ncclGinCounter_t counter, uint64_t least,
                 int bits = 56, cuda::memory_order ord = cuda::memory_order_acquire)
```

**resetCounter**: Reset counter to zero
```cpp
void resetCounter(ncclGinCounter_t counter)
```

### GIN Barriers

#### ncclGinBarrierSession
**Purpose**: Network barrier synchronization across GIN-accessible peers.

**Constructors**:
```cpp
// Rail-scoped barrier
ncclGinBarrierSession(Coop coop, ncclGin gin, ncclTeamTagRail tag, uint32_t index)

// General-purpose barrier
ncclGinBarrierSession(Coop coop, ncclGin gin, ncclTeam team,
                      ncclGinBarrierHandle handle, uint32_t index)
```

**sync Method**:
```cpp
void sync(Coop coop, cuda::memory_order order, ncclGinFenceLevel fence)
```

**Parameters**:
- `order`: Memory ordering (`cuda::memory_order_relaxed`, `cuda::memory_order_release`, etc.)
- `fence`: Fence level for GIN operations
  - `ncclGinFenceLevel::Relaxed`: The only fence level in NCCL 2.28.9

### Usage Patterns

#### Basic GIN Put with Signal
```cpp
ncclTeam world = ncclTeamWorld(devComm);
ncclGin gin { devComm, 0 };  // context 0

// Read initial signal value
uint64_t signalValue = gin.readSignal(signalIndex);

// Issue puts to remote peers with signal increment
for (int r = 0; r < world.nRanks; ++r) {
    if (r != world.rank) {
        gin.put(world, r,
                recvwin, recvOffset + world.rank * bytes,
                sendwin, sendOffset + r * bytes,
                bytes,
                ncclGin_SignalInc{signalIndex});
    }
}

// Wait for all remote peers to complete their puts
gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + numRemotePeers);

// Flush to ensure source buffers can be reused
gin.flush(ncclCoopCta());
```

#### GIN Barrier Pattern (World team — general constructor)
```cpp
// World-team barrier requires the general constructor (ncclTeam + ncclGinBarrierHandle).
// There is NO convenience constructor for ncclTeamTagWorld; only ncclTeamTagRail has one.
ncclGinBarrierSession<ncclCoopCta> bar {
    ncclCoopCta(),
    gin,                          // ncclGin instance (2nd arg)
    ncclTeamWorld(devComm),       // ncclTeam (3rd arg)
    devComm.railGinBarrier,       // ncclGinBarrierHandle (4th arg)
    blockIdx.x                    // barrier index (5th arg)
};

// Initial barrier with relaxed ordering
bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

// Perform GIN operations...

// Final barrier with release semantics
bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
```

#### GIN Barrier Pattern (Rail team — convenience constructor)
```cpp
ncclGinBarrierSession<ncclCoopCta> bar {
    ncclCoopCta(),
    gin,                          // ncclGin instance
    ncclTeamTagRail{},            // convenience tag
    blockIdx.x                    // barrier index
};
bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);
```

### Setup Requirements (Host-Side)

Enable GIN support during communicator creation:
```cpp
ncclDevCommRequirements reqs;
memset(&reqs, 0, sizeof(reqs));
reqs.railGinBarrierCount = /* number needed */;
reqs.ginSignalCount = /* number needed */;
reqs.ginCounterCount = /* number needed */;
ncclDevCommCreate(comm, &reqs, &devComm);
(void)cudaGetLastError();  // MUST clear stale CUDA error left by ncclDevCommCreate
```

GIN is available when the NCCL communicator connects ranks over InfiniBand/RoCE.
For intra-node PCIe A100 GPUs, GIN uses GDAKI (GPU Direct Async Kernel-Initiated).
"""


# Code-level examples adapted from NVIDIA NCCL Device API Pure GIN AlltoAll example:
# https://github.com/NVIDIA/nccl/blob/master/examples/06_device_api/02_alltoall_gin/main.cu
# https://github.com/NVIDIA/nccl/blob/master/examples/06_device_api/02_alltoall_gin/README.md
NCCL_GIN_PURE_EXAMPLE = """## NCCL Pure GIN Code Examples (from official AlltoAll GIN example)

These patterns are taken from the NVIDIA NCCL repo `examples/06_device_api/02_alltoall_gin`.
Use them when evolving GIN kernels for network-based, device-initiated communication.

### Host-side: Device communicator creation (GIN only)

CTA count must match railGinBarrierCount for proper barrier synchronization.

```cpp
#define NCCL_DEVICE_CTA_COUNT 1
#define NCCL_DEVICE_THREADS_PER_CTA 512

ncclDevComm devComm;
ncclDevCommRequirements reqs;
memset(&reqs, 0, sizeof(reqs));
reqs.railGinBarrierCount = NCCL_DEVICE_CTA_COUNT;  // GIN barriers for network sync
reqs.ginSignalCount = 1;                            // GIN signals for completion
NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
(void)cudaGetLastError();  // MUST clear stale CUDA error left by ncclDevCommCreate
```

### Host-side: Symmetric memory and window registration

Device API requires symmetric memory; use ncclMemAlloc and NCCL_WIN_COLL_SYMMETRIC.

```cpp
void* d_sendbuff;
void* d_recvbuff;
ncclWindow_t send_win;
ncclWindow_t recv_win;

NCCLCHECK(ncclMemAlloc(&d_sendbuff, size_bytes));
NCCLCHECK(ncclMemAlloc(&d_recvbuff, size_bytes));

NCCLCHECK(ncclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win, NCCL_WIN_COLL_SYMMETRIC));
NCCLCHECK(ncclCommWindowRegister(comm, d_recvbuff, size_bytes, &recv_win, NCCL_WIN_COLL_SYMMETRIC));
```

### Device-side: GIN barrier (coordinate across ranks over network)

GIN barriers enable cross-node synchronization from device code. Each block uses blockIdx.x
as its barrier index so blocks can progress independently while coordinating with other nodes.

```cpp
ncclGin gin { devComm, 0 };
ncclGinBarrierSession<ncclCoopCta> bar {
    ncclCoopCta(),
    gin,
    ncclTeamWorld(devComm),
    devComm.railGinBarrier,
    blockIdx.x
};
bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);
```

### Device-side: readSignal before puts, waitSignal after (rolling value)

Read initial signal value, issue puts that increment the remote signal, then wait until
signal reaches initialValue + number of expected increments.

```cpp
unsigned int signalIndex = 0;
uint64_t signalValue = gin.readSignal(signalIndex);

// Issue puts (each put can use ncclGin_SignalInc{signalIndex})
for (int r = tid; r < devComm.nRanks; r += nthreads) {
    gin.put(ncclTeamWorld(devComm), r,
            recvwin, recvoffset + devComm.rank * size,
            sendwin, sendoffset + r * size,
            size, ncclGin_SignalInc{signalIndex});
}

// Wait for all remote puts to complete (rolling comparison handles overflow)
gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);
gin.flush(ncclCoopCta());
```

### Device-side: Full pure GIN kernel sketch (AlltoAll-style)

```cpp
template <typename T>
__global__ void PureGinAlltoAllKernel(ncclWindow_t sendwin, size_t sendoffset,
                                      ncclWindow_t recvwin, size_t recvoffset,
                                      size_t count, ncclDevComm devComm) {
    int ginContext = 0;
    unsigned int signalIndex = 0;
    ncclGin gin { devComm, ginContext };
    uint64_t signalValue = gin.readSignal(signalIndex);

    ncclGinBarrierSession<ncclCoopCta> bar {
        ncclCoopCta(), gin, ncclTeamWorld(devComm),
        devComm.railGinBarrier, blockIdx.x
    };
    bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int nthreads = blockDim.x * gridDim.x;
    const size_t size = count * sizeof(T);

    for (int r = tid; r < devComm.nRanks; r += nthreads) {
        gin.put(ncclTeamWorld(devComm), r,
                recvwin, recvoffset + devComm.rank * size,
                sendwin, sendoffset + r * size,
                size, ncclGin_SignalInc{signalIndex});
    }

    gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);
    gin.flush(ncclCoopCta());
}
```

### When to use pure GIN (from official README)

- Communication between ranks that cannot use LSA (e.g. different nodes)
- Network-based collectives in multi-node environments
- All communication must go through the network

### Performance notes (from official README)

- Signal-based completion enables asynchronous operation patterns
- Multiple GIN contexts can improve parallel communication performance
- CTA count must match railGinBarrierCount for proper barrier synchronization
"""


NCCL_THREAD_GROUPS_DOC = """## NCCL Thread Groups (Cooperative Groups)

### Overview
Thread groups specify which threads within a CTA participate in NCCL operations.
Many device API functions take a cooperative group parameter.

### Predefined Thread Groups

**ncclCoopThread()** - Single thread participation
**ncclCoopWarp()** - Warp-level (32 threads)
**ncclCoopCta()** - Block-level (all threads) - **Most commonly used**

### Usage Patterns

#### Full CTA Synchronization (Most Common)
```cpp
// Create barrier with ncclCoopCta
ncclLsaBarrierSession<ncclCoopCta> lsaBar {
    ncclCoopCta(), devComm, ncclTeamLsa(devComm),
    devComm.lsaBarrier, blockIdx.x
};

// All threads synchronize
lsaBar.sync(ncclCoopCta(), cuda::memory_order_release);
gin.flush(ncclCoopCta());
```

#### Leader Thread Pattern
```cpp
// Only thread 0 issues operation
if (threadIdx.x == 0) {
    gin.put(world, peer, dstWnd, dstOffset, srcWnd, srcOffset, bytes,
            ncclGin_SignalInc{signalIndex}, {}, ncclCoopThread());
}
// All threads wait
gin.waitSignal(ncclCoopCta(), signalIndex, expectedValue);
```

#### Warp-Level Operations
```cpp
// All 32 threads in each warp must participate when using ncclCoopWarp.
// Do NOT guard with `if (threadIdx.x % 32 == 0)` — that would leave
// only lane 0, which contradicts the warp cooperative group contract.
int warpId = threadIdx.x / 32;
gin.put(world, peer + warpId, dstWnd, dstOffset,
        srcWnd, srcOffset, bytes,
        ncclGin_SignalInc{signalIndex}, ncclGin_None{}, ncclCoopWarp());
```

### Custom Thread Groups

Implement custom groups by providing three methods:
- `thread_rank()`: Thread identifier within group
- `size()`: Total threads in group
- `sync()`: Synchronization mechanism

NCCL is also compatible with CUDA cooperative groups (`cooperative_groups::thread_block`, etc.).

### Guidelines

- **ncclCoopCta()**: Default choice for most operations, full block synchronization
- **ncclCoopWarp()**: For warp-granular operations, performance optimization
- **ncclCoopThread()**: Single thread operations, typically with leader thread pattern
"""


NCCL_TEAMS_DOC = """## NCCL Teams

### Overview
Teams represent subsets of ranks within an NCCL communicator for scoped communication.

### Predefined Teams

**ncclTeamWorld(devComm)** - All ranks in the communicator (global scope)
**ncclTeamLsa(devComm)** - Ranks accessible via load/store operations (local GPU group)
**ncclTeamRail(devComm)** - Ranks directly accessible over network (rail topology)

### Team Structure

The `ncclTeam` type contains:
- `nRanks`: Number of ranks in the team
- `rank`: Current rank position within the team (0-indexed)
- `stride`: Spacing between ranks

**Properties**:
- World and LSA teams are contiguous (stride = 1)
- Rail teams are typically non-contiguous (stride = LSA team size)

### Usage Patterns

#### Accessing Team Properties
```cpp
ncclTeam world = ncclTeamWorld(devComm);
ncclTeam lsa = ncclTeamLsa(devComm);
ncclTeam rail = ncclTeamRail(devComm);

// Use team properties
for (int r = 0; r < world.nRanks; ++r) {
    // Communicate with rank r in world team
}

for (int lp = 0; lp < lsa.nRanks; ++lp) {
    // Access LSA peer lp
    float* peer = (float*)ncclGetLsaPointer(window, offset, lp);
}
```

#### Team-Scoped GIN Operations
```cpp
ncclTeam world = ncclTeamWorld(devComm);

// Put to specific peer in world team
gin.put(world, peerRank, dstWnd, dstOffset, srcWnd, srcOffset, bytes,
        ncclGin_SignalInc{signalIndex});
```

#### Hybrid LSA + GIN Pattern
```cpp
ncclTeam world = ncclTeamWorld(devComm);
ncclTeam lsa = ncclTeamLsa(devComm);

// The LSA team is contiguous in world rank space.
// world.rank is this GPU's world rank; lsa.rank is this GPU's position
// within the LSA team. The first world rank in the LSA group is:
int lsaWorldStart = world.rank - lsa.rank;

// Iterate over world, distinguish LSA vs non-LSA peers
for (int r = 0; r < world.nRanks; ++r) {
    if (r >= lsaWorldStart && r < lsaWorldStart + lsa.nRanks) {
        // Use LSA for this peer (convert world rank → LSA rank)
        int lsaPeer = r - lsaWorldStart;
        float* peer = (float*)ncclGetLsaPointer(recvwin, offset, lsaPeer);
        peer[idx] = data;
    } else {
        // Use GIN for remote peer
        gin.put(world, r, recvwin, offset, sendwin, offset, bytes,
                ncclGin_SignalInc{signalIndex});
    }
}
```

### Guidelines

- **World team**: Use for global all-to-all communication or when addressing all ranks
- **LSA team**: Use for direct memory access within local GPU group
- **Rail team**: Use for optimized network topology communication (advanced)
"""


# ---------------------------------------------------------------------------
# Actual NCCL device header files (authoritative API surface)
# ---------------------------------------------------------------------------

NCCL_HEADER_GIN_H = r"""## nccl_device/gin.h — GIN Session API (authoritative header)

```cpp
// ncclGin (alias for ncclGin_BackendMask<NCCL_GIN_BACKEND_MASK_ALL>)
//
// Construct with: ncclGin gin(devComm, contextIndex);
//   contextIndex is typically 0 (must be < ginContextCount in requirements).

// Completion action types for put():
struct ncclGin_None {};                         // No remote action / no local action
struct ncclGin_SignalAdd { ncclGinSignal_t signal; uint64_t value; };
struct ncclGin_SignalInc { ncclGinSignal_t signal; };   // Equivalent to SignalAdd{+1}
// SignalInc: for a given signal, all operations between successive reset()'s
// must either all be SignalInc or all not SignalInc.
struct ncclGin_CounterInc { ncclGinCounter_t counter; };
struct ncclGin_DescriptorSmem { ncclGinDescriptorSmem* descriptor; };

// ---- put (window+offset variant) ----
// Puts bytes from local srcWnd+srcOffset to peer's dstWnd+dstOffset.
// RemoteAction: signal action on the *receiver* when put completes. Signal is
//   visible only after this put AND all preceding puts on this context to the
//   same peer have settled.
// LocalAction: action locally when source buffer has been consumed.
// Coop: set of threads participating (default ncclCoopThread).
template<
  typename RemoteAction = ncclGin_None,
  typename LocalAction = ncclGin_None,
  typename Coop = ncclCoopThread,
  typename DescriptorSmem = ncclGin_None>
void put(
  ncclTeam, int peer,
  ncclWindow_t dstWnd, size_t dstOffset,
  ncclWindow_t srcWnd, size_t srcOffset, size_t bytes,
  RemoteAction = ncclGin_None{},
  LocalAction = ncclGin_None{},
  Coop = ncclCoopThread{},
  DescriptorSmem = ncclGin_None{},
  cuda::thread_scope alreadyReleased = cuda::thread_scope_thread,
  cuda::thread_scope expected_scope = cuda::thread_scope_device) const;

// ---- put (ncclSymPtr variant) ----
template<typename T, ...same template args...>
void put(ncclTeam, int peer,
  ncclSymPtr<T> dstElts, ncclSymPtr<T> srcElts, size_t nElts, ...) const;

// ---- putValue (write a scalar <= 8 bytes to remote) ----
template<typename T, typename RemoteAction=ncclGin_None, typename Coop=ncclCoopThread, ...>
void putValue(ncclTeam, int peer,
  ncclWindow_t dstWnd, size_t dstOffset, T value, ...) const;

// ---- signal (signal without data) ----
template<typename RemoteAction, typename Coop=ncclCoopThread, ...>
void signal(ncclTeam, int peer, RemoteAction, Coop = ncclCoopThread(), ...) const;

// ---- flush ----
// All source buffers from put's from any thread in this coop will be safe to reuse.
// Flush does NOT guarantee that data has settled in remote memory.
// IMPORTANT: The Coop must match the threads that actually issued the puts.
//   e.g. if only thread (0,0) issued puts, use ncclCoopThread, NOT ncclCoopCta.
template<typename Coop>
void flush(Coop, cuda::memory_order ord = cuda::memory_order_acquire) const;

// ---- Signal/Counter wait ----
// Uses "rolling" comparison: rolling_less_equal(supplied, internal).
// Counters use max 56 bits. Signals use 64 bits by default.

uint64_t readCounter(ncclGinCounter_t counter, int bits=56, ...) const;

template<typename Coop>
void waitCounter(Coop, ncclGinCounter_t counter, uint64_t least, int bits=56, ...) const;

// Signal shadow: user-manipulable mirror of signal value.
uint64_t* getSignalShadowPtr(ncclGinSignal_t signal) const;
void increaseSignalShadow(ncclGinSignal_t signal, uint64_t delta) const;

uint64_t readSignal(ncclGinSignal_t signal, int bits=64, ...) const;

// Wait for signal to meet or exceed 'least'.
template<typename Coop>
void waitSignal(Coop, ncclGinSignal_t signal, uint64_t least, int bits=64, ...) const;

// Wait for signal to meet or exceed shadow value.
template<typename Coop>
void waitSignalMeetShadow(Coop, ncclGinSignal_t signal, int bits=64, ...) const;

// Wait until signal exceeds shadow by leastDelta, updates shadow, returns before/delta.
template<typename Coop, typename Uint>
void waitSignalFollowShadow(Coop, ncclGinSignal_t signal, Uint leastDelta,
  Uint* before, Uint* delta, int bits=64, ...) const;

// Reset (sets to zero). May NOT race with concurrent modifications.
void resetCounter(ncclGinCounter_t counter) const;
void resetSignal(ncclGinSignal_t signal) const;
```
"""


NCCL_HEADER_CORE_H = r"""## nccl_device/core.h — Core types, teams, windows, ncclDevComm

```cpp
// Forward declarations
struct ncclDevComm;
struct ncclTeam;
// ncclWindow_t is defined in nccl.h

typedef uint32_t ncclGinSignal_t;
typedef uint32_t ncclGinCounter_t;

// ---- ncclTeam ----
struct ncclTeam {
  int nRanks, rank, stride;
};

// ---- ncclDevCommRequirements ----
// Passed to ncclDevCommCreate to configure the device communicator.
struct ncclDevCommRequirements {
  ncclDevResourceRequirements_t* resourceRequirementsList;
  ncclTeamRequirements_t* teamRequirementsList;

  bool lsaMultimem;

  int barrierCount;
  int lsaBarrierCount;
  int railGinBarrierCount;      // Typically 1

  int lsaLLA2ABlockCount, lsaLLA2ASlotCount;

  bool ginForceEnable;
  int ginContextCount;          // Hint; typically 1
  int ginSignalCount;           // Must be >= number of distinct signal indices used
  int ginCounterCount;          // Must be >= number of distinct counter indices used
};

// Host-side create/destroy:
ncclResult_t ncclDevCommCreate(ncclComm_t, ncclDevCommRequirements_t const*, ncclDevComm_t* outDevComm);
ncclResult_t ncclDevCommDestroy(ncclComm_t, ncclDevComm_t const* devComm);

// ---- Team API ----
ncclTeam ncclTeamWorld(ncclDevComm const&);   // All ranks
ncclTeam ncclTeamLsa(ncclDevComm const&);     // Local (NVLink) ranks
ncclTeam ncclTeamRail(ncclDevComm const&);    // Network rail ranks

int ncclTeamRankToWorld(ncclDevComm const&, ncclTeam, int rank);
int ncclTeamRankToLsa(ncclDevComm const&, ncclTeam, int rank);
bool ncclTeamRankIsMember(ncclTeam_t a, ncclTeam_t b, int bPeer);
int ncclTeamRankToTeam(ncclTeam_t a, ncclTeam_t b, int bPeer);
int ncclTeamRankInDifference(ncclTeam_t parent, ncclTeam_t subset, int index);
ncclTeam_t ncclTeamInnerFactor(ncclTeam_t parent, int innerSize);
ncclTeam_t ncclTeamOuterFactor(ncclTeam_t parent, int innerSize);

// ---- Window API ----
// Get local pointer into a registered window at byte offset:
void* ncclGetLocalPointer(ncclWindow_t w, size_t offset);
// Get peer's pointer (requires LSA connectivity):
void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, int peer);
void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, ncclTeam tm, int peer);
// LSA peer pointer:
void* ncclGetLsaPointer(ncclWindow_t w, size_t offset, int peer);
```
"""


NCCL_HEADER_COOP_H = r"""## nccl_device/coop.h — Cooperative group types

```cpp
// NCCL cooperative group types (used as Coop template parameters for GIN calls).
// Each type provides: thread_rank(), size(), num_threads(), sync().

// ncclCoopThread — single thread (default for put/waitSignal)
typedef ncclCoopTile<1> ncclCoopThread;

// ncclCoopWarp — full 32-thread warp
typedef ncclCoopTile<32> ncclCoopWarp;

// ncclCoopCta — entire CTA (threadblock)
struct ncclCoopCta {
  int thread_rank() const { return threadIdx.x; }
  int size() const { return blockDim.x; }
  void sync() { __syncthreads(); }
};

// ncclCoopTile<N> — aligned power-of-2 set of threads within a warp (N <= 32)
template<int nThreadsPow2>
struct ncclCoopTile {
  int thread_rank() const;  // lane % nThreadsPow2
  int size() const;         // nThreadsPow2
  void sync();              // __syncwarp(laneMask)
};

// ncclCoopLanes — arbitrary subset of lanes in a warp
struct ncclCoopLanes {
  uint32_t lmask;
  int thread_rank() const;  // popcount of lower lanes
  int size() const;         // popcount(lmask)
  void sync();              // __syncwarp(lmask)
};

// ncclCoopWarpSpan — consecutive warps with unique id [0..15]
struct ncclCoopWarpSpan {
  uint32_t warp0:8, nWarps:8, id:8;
  int thread_rank() const;  // threadIdx.x - 32*warp0
  int size() const;         // 32*nWarps
  void sync();              // __barrier_sync_count(1+id, 32*nWarps)
};

// Helper: get active/coalesced lanes
ncclCoopLanes ncclCoopCoalesced();  // __activemask()

// IMPORTANT usage notes:
// - flush(Coop): Coop must match the threads that issued puts.
//   If only one thread issued puts, use ncclCoopThread, NOT ncclCoopCta.
// - waitSignal(Coop, ...): ncclCoopThread is safest; ncclCoopCta can deadlock
//   if not all CTA threads reach the wait.
```
"""


NCCL_HEADER_PTR_H = r"""## nccl_device/ptr.h — Symmetric pointer type

```cpp
// ncclSymPtr<T>: a window + byte offset pair for type-safe symmetric addressing.
// Can be used with the ncclSymPtr variant of gin.put().
template<typename T>
struct ncclSymPtr {
  ncclWindow_t window;
  size_t offset;

  ncclSymPtr(ncclWindow_t window = nullptr, size_t offset = 0);

  // Arithmetic
  ncclSymPtr<T>& operator+=(int/unsigned/long/...);
  ncclSymPtr<T>& operator-=(int/unsigned/long/...);

  // Device-side pointer resolution
  T* localPtr() const;          // Local pointer
  T* lsaPtr(int peer) const;    // LSA peer pointer
  T* peerPtr(int peer) const;   // Peer pointer
  T* peerPtr(ncclTeam team, int peer) const;
  T* multimemPtr(ncclMultimemHandle mmHandle) const;
  T* lsaMultimemPtr(ncclDevComm const&) const;
};
```
"""


NCCL_HOST_TO_DEVICE_COOKBOOK = r"""## GIN Kernel Signature & Host Launch Pattern (from verified working reference)

### GIN Kernel Signature

The GIN kernel receives ncclDevComm BY VALUE (not by pointer):
```cpp
__global__ void ginKernel(
    ncclDevComm devComm,           // BY VALUE — this is correct
    ncclWindow_t local_win,        // source window (registered over local buffer)
    ncclWindow_t remote_win,       // destination window (registered over remote receive buffer)
    int* C_local,                  // local data pointer (for accumulation after transfer)
    int* C_remote,                 // remote receive buffer pointer
    int row_offset, int chunk_rows,
    int chunk_id)                  // used for rolling signal counting
{
    ncclGin gin(devComm, 0);       // context 0
    int target_rank = 1 - devComm.rank;  // peer rank (2-GPU case)

    // Only leader thread issues put
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        size_t chunk_bytes = (size_t)chunk_rows * N * sizeof(int);
        size_t offset_bytes = (size_t)row_offset * N * sizeof(int);

        gin.put(ncclTeamWorld(devComm), target_rank,
                remote_win, offset_bytes,     // dst: peer's remote_win
                local_win,  offset_bytes,     // src: my local_win
                chunk_bytes,
                ncclGin_SignalInc{0});         // increment signal index 0

        gin.flush(ncclCoopThread());  // MUST match the threads that issued put
    }

    // ALL threads wait for remote data (rolling signal: chunk_id + 1)
    gin.waitSignal(ncclCoopThread(), 0, chunk_id + 1);

    // Accumulate: C_local += C_remote
    int local_row  = blockIdx.y * blockDim.y + threadIdx.y;
    int global_row = row_offset + local_row;
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_row < row_offset + chunk_rows && global_col < N) {
        int idx = global_row * N + global_col;
        C_local[idx] += C_remote[idx];
    }
}
```

### Host-side Launch (inside chunk loop)

```cpp
for (int i = 0; i < NUM_CHUNKS; i++) {
    int offset = i * rows_per_chunk;
    matMulCompute<<<grid, block, 0, stream>>>(d_A, d_B, d_C_local, offset, rows_per_chunk);
    ginKernel<<<grid, block, 0, stream>>>(
        d_comm, local_win, remote_win, d_C_local, d_C_remote,
        offset, rows_per_chunk, i);
}
```

### Key facts
- `ncclDevComm` is passed by value, NOT pointer
- `ncclGin_SignalInc{0}` — field is `ncclGinSignal_t` (uint32_t), 0 is signal index
- Rolling signal: chunk i waits for signal >= i+1
- `gin.flush(ncclCoopThread())` after leader-thread put (Coop must match who issued puts)
- `gin.waitSignal(ncclCoopThread(), ...)` — each thread waits independently
- After ncclDevCommCreate on host, call `(void)cudaGetLastError()` to clear stale error
"""


NCCL_HEADER_GIN_BARRIER_H = r"""## nccl_device/gin_barrier.h — GIN barrier API

```cpp
// GIN fence levels
enum class ncclGinFenceLevel {
  Relaxed
};

// Host-side: create barrier requirement
ncclResult_t ncclGinBarrierCreateRequirement(
  ncclComm_t, ncclTeam_t, int nBarriers,
  ncclGinBarrierHandle_t* outHandle,
  ncclDevResourceRequirements_t* outReq);

// Device-side barrier session
template<typename Coop>
struct ncclGinBarrierSession {
  // Constructors
  ncclGinBarrierSession(Coop, ncclGin, ncclTeam, ncclGinBarrierHandle, uint32_t index);
  ncclGinBarrierSession(Coop, ncclGin, ncclTeamTagRail, uint32_t index);

  // Not copyable
  ncclGinBarrierSession(ncclGinBarrierSession const&) = delete;

  // Synchronize all team members
  void sync(Coop, cuda::memory_order, ncclGinFenceLevel);
};
```
"""


NCCL_HEADER_BARRIER_H = r"""## nccl_device/barrier.h — Combined LSA+GIN barrier API

```cpp
// ncclBarrierSession combines LSA and GIN barriers into a single session.
// Use this when you need to synchronize across both LSA and GIN peers
// (e.g., world-team barrier spanning local NVLink + remote network).

template<typename Coop>
struct ncclBarrierSession {
  // Full featured constructor:
  ncclBarrierSession(Coop, ncclTeam innerTeam, ncclTeam outerTeam, ncclGin,
    ncclLsaBarrierHandle innerBarHandle,
    ncclGinBarrierHandle outerBarHandle,
    uint32_t index,
    bool multimem = false, ncclMultimemHandle innerMmHandle = {});

  // Convenience constructors for baked-in teams:
  ncclBarrierSession(Coop, ncclTeamTagWorld, ncclGin, uint32_t index, bool multimem = false);
  ncclBarrierSession(Coop, ncclTeamTagLsa, ncclDevComm const&, uint32_t index, bool multimem = false);
  ncclBarrierSession(Coop, ncclTeamTagRail, ncclGin, uint32_t index);

  ncclBarrierSession(ncclBarrierSession const&) = delete;  // Not copyable

  // Access sub-barriers
  ncclLsaBarrierSession<Coop>& lsaBarrier();
  ncclGinBarrierSession<Coop>& ginBarrier();

  // Synchronize all team members (both LSA and GIN)
  void sync(Coop, cuda::memory_order, ncclGinFenceLevel);
};
```
"""
