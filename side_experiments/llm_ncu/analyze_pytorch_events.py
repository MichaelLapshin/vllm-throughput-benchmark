import json
import os
import pathlib
import math
from dataclasses import dataclass
from typing import Tuple, List

CURRENT_DIR = pathlib.Path(__file__).parent.absolute()

@dataclass
class KernelFuncArgs:
    queued: int
    correlation: int
    registers_per_thread: int
    shared_memory: int
    blocks_per_sm: float
    warps_per_sm: float
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]
    est_occupancy_percentage: int

    def get_num_warps(self) -> int:
        return (math.ceil((self.block[0] / THREADS_PER_WARP) * \
                          math.ceil(self.block[1] / THREADS_PER_WARP) * \
                          math.ceil((self.block[2] / THREADS_PER_WARP)))) * \
            (self.grid[0] * self.grid[1] * self.grid[2])
    
    def get_num_threads(self) -> int:
        return (self.block[0] * self.block[1] * self.block[2]) * \
            (self.grid[0] * self.grid[1] * self.grid[2])


@dataclass
class KernelEvent:
    ph: str
    cat: str
    name: str
    ts: float
    dur: float
    args: KernelFuncArgs

THREADS_PER_WARP = 32

def parse_pytorch_events(file_path: str):
    if not os.path.exists(file_path):
        raise ValueError(f"Error: The file '{file_path}' was not found.")

    with open(file_path) as file:
        data = json.load(file)
        return data["traceEvents"]

def extract_kernel_events(file_path: str) -> List[KernelEvent]:
    events = parse_pytorch_events(file_path)
    kernel_events = list(filter(
        lambda d: "cat" in d and d["cat"] == "kernel",
        events
    ))
    return [
        KernelEvent(
            ph=k["ph"],
            cat=k["cat"],
            name=k["name"],
            ts=k["ts"],
            dur=k["dur"],
            # args
            args=KernelFuncArgs(
                queued=k["args"]["queued"],
                correlation=k["args"]["correlation"],
                registers_per_thread=k["args"]["registers per thread"],
                shared_memory=k["args"]["shared memory"],
                blocks_per_sm=k["args"]["blocks per SM"],
                warps_per_sm=k["args"]["warps per SM"],
                grid=k["args"]["grid"],
                block=k["args"]["block"],
                est_occupancy_percentage=k["args"]["est. achieved occupancy %"],
            )
        )
        for k in kernel_events
    ]

def count_max_overlap(events, event_name: str) -> int:
    # Gather points
    points = []
    for e in events:
        if e.name == event_name:
            points.append((e.ts, 1))
            points.append((e.ts + e.dur, -1))
    points.sort()

    # Check overlap
    max_overlaps = 0
    current_overlaps = 0
    for _, change in points:
        current_overlaps += change
        max_overlaps = max(max_overlaps, current_overlaps)
    return max_overlaps

def print_kernel_events_summary(file_path: str, output_path: str = ""):
    kevents = extract_kernel_events(file_path)

    # Organize per-funtion arguments
    name_args_map = {}
    for e in kevents:
        name_args_map.setdefault(e.name, []).append(e)
    
    # Summarize
    out = f"=== '{file_path}' ===\n"
    out += f"\n+++ Summary +++\n"
    out += f"    Total Duration: {sum(e.dur for e in kevents)}\n"
    out += f"    Total # Calls: {len(kevents)}\n"
    out += f"    Total # Warps: {sum(e.args.get_num_warps() for e in kevents)}\n"
    out += f"    Total # Threads: {sum(e.args.get_num_threads() for e in kevents)}\n"
    out += f"\n+++ Functions +++\n"
    for name, events in sorted(name_args_map.items()):
        args = [e.args for e in events]
        out += f"{name}\n"
        out += f"    Total Duration: {sum(e.dur for e in events)}\n"
        out += f"    Max Overlap: {count_max_overlap(events, name)}\n"
        out += f"    Total # Calls: {len(events)}\n"
        out += f"    Total # Warps: {sum(a.get_num_warps() for a in args)}\n"
        out += f"    Total # Threads: {sum(a.get_num_threads() for a in args)}\n"
        out += f"    Total # Shared Memory: {sum(a.shared_memory for a in args)}\n"
        out += f"    Avg Occupancy %: {sum(e.dur * e.args.est_occupancy_percentage for e in events)/sum(e.dur for e in events)}\n"

        grid_shape = events[0].args.grid
        if all(grid_shape == e.args.grid for e in events):
            out += f"    Grid Shape: {grid_shape}\n"
        else:
            grids = {}
            for e in events:
                grid = tuple(e.args.grid)
                if grid not in grids:
                    grids[grid] = 0
                grids[grid] += 1
            out += f"    Diff Grids: {grids}\n"
        
        block_shape = events[0].args.block
        if all(block_shape == e.args.block for e in events):
            out += f"    Block Shape: {block_shape}\n"
        else:
            blocks = {}
            for e in events:
                block = tuple(e.args.block)
                if block not in blocks:
                    blocks[block] = 0
                blocks[block] += 1
            out += f"    Diff Block: {blocks}\n"

    # Output contents
    if output_path:
        with open(output_path, "w") as file:
            file.write(out)
    else:
        print(out)

def compare_kernel_traces():
    base_dir = f"{CURRENT_DIR}/results/Qwen3-0_6B-Feb21/NoSpecDecScheduler_Batched"
    print_kernel_events_summary(
        f"{base_dir}/1771718959231635296-rank-0.1771718982607801449.pt.trace.json",
        "out1.txt"
    )
    print_kernel_events_summary(
        f"{base_dir}/1771718983733919277-rank-0.1771719007270792140.pt.trace.json",
        "out2.txt"
    )

    print_kernel_events_summary(
        f"{base_dir}/1771719008478231837-rank-0.1771719031867716436.pt.trace.json",
        "out3.txt"
    )

    spec_dir = f"{CURRENT_DIR}/results/Qwen3-0_6B-Feb21/VerifySuccessScheduler"
    print_kernel_events_summary(
        f"{spec_dir}/1771718506387870623-rank-0.1771718528245972273.pt.trace.json",
        "spec1.txt"
    )
    print_kernel_events_summary(
        f"{spec_dir}/1771718529639340939-rank-0.1771718550883198929.pt.trace.json",
        "spec2.txt"
    )
    print_kernel_events_summary(
        f"{spec_dir}/1771718552190822963-rank-0.1771718573372685462.pt.trace.json",
        "spec3.txt"
    )
    print_kernel_events_summary(
        f"{spec_dir}/1771718597184064767-rank-0.1771718618393094686.pt.trace.json",
        "spec4.txt"
    )
    print_kernel_events_summary(
        f"{spec_dir}/1771718619711827336-rank-0.1771718640899956525.pt.trace.json",
        "spec5.txt"
    )

    # Spec Batched (2 reqs)
    spec_dir_8b = f"{CURRENT_DIR}/results/Qwen3-8B-Feb21/VerifySuccessScheduler"
    print_kernel_events_summary(
        f"{spec_dir_8b}/1771741912398859444-rank-0.1771741940566267289.pt.trace.json",
        "spec2_8b.txt"
    )
    print_kernel_events_summary(
        f"{spec_dir_8b}/1771741971880585103-rank-0.1771742000222147225.pt.trace.json",
        "spec4_8b.txt"
    )

    spec_batch_dir_8b = f"{CURRENT_DIR}/results/Qwen3-8B-Feb21/VerifySuccessScheduler_Batched"
    print_kernel_events_summary(
        f"{spec_batch_dir_8b}/1771785439305418227-rank-0.1771785467795340979.pt.trace.json",
        "spec_batch2.txt"
    )
    print_kernel_events_summary(
        f"{spec_batch_dir_8b}/1771785468375615502-rank-0.1771785496563104673.pt.trace.json",
        "spec_batch4.txt"
    )

if __name__ == "__main__":
    # Run the program
    compare_kernel_traces()
