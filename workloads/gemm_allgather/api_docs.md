## NCCL Device API Overview

Host-side setup is complete. Focus on device-kernel usage only.

**Core Concepts:**
- **LSA (Load/Store Accessible)**: Direct memory access to local GPU peers
- **GIN (GPU-Initiated Networking)**: One-sided network transfers to remote GPUs
- **Teams**: Subsets of ranks - World (all), LSA (local), Rail (network)
- **Thread Groups**: Block (`ncclCoopCta`), Warp (`ncclCoopWarp`), Thread (`ncclCoopThread`)

**Key Principles:**
- Only use device APIs inside kernels (no host-side calls)
- Synchronize appropriately between writes and reads


## NCCL LSA (Load/Store Accessible) API Documentation

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


## NCCL Thread Groups (Cooperative Groups)

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


## NCCL Teams

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


## nccl_device/core.h — Core types, teams, windows, ncclDevComm

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


## nccl_device/coop.h — Cooperative group types

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


## nccl_device/ptr.h — Symmetric pointer type

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


## nccl_device/barrier.h — Combined LSA+GIN barrier API

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
