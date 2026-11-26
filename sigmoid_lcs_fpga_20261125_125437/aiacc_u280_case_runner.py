#!/usr/bin/env python3
"""
Run one AIACC load_store_v3 / v4.21 multi_inst_stage case on a U280 XDMA image.

This is a host-side runner. It mirrors the RTL multi-stage TB at software level:
  1) write memory_init{1..8}.hex to HBM windows
  2) load each stage's iq_inst_*.txt through /dev/xdma1_user BAR offsets
  3) write SMI IQ_EN and START registers, poll INT/DONE, ack INT
  4) read back store window from HBM and compare with output_bank golden

The stage data_list remains authoritative for IQ load order.  Its absolute paths
may be stale after packaging, so only each listed file's basename is resolved
inside the packaged stage directory.
"""

import argparse
import json
import mmap
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CSR_CFG = 0x000
CSR_INT = 0x004
CSR_IQ_EN = [0x010, 0x014, 0x018, 0x01C]
CSR_START = 0x020
CSR_DONE = 0x024
CSR_RTL_VERSION = 0x080
EXPECTED_RTL_VERSION = 0x00042200
DEFAULT_HBM_STRIDE = 0x10000000
DEFAULT_BAR_SIZE = 0x10000
DEFAULT_CSR_SETTLE_S = 0.001
AXI_LITE_BASE = 0x80000000

PACKER_DATA = [0xF00, 0xF04, 0xF08, 0xF0C]
PACKER_WSTRB = 0xF10
PACKER_IQADDR = 0xF14
PACKER_CTRL = 0xF18
PACKER_STATUS = 0xF1C
PACKER_CAP = 0xF20
PACKER_CAP_IQP1 = 0x49515031

PACKER_CTRL_COMMIT = 1 << 0
PACKER_CTRL_CLEAR_ERROR = 1 << 1
PACKER_CTRL_CLEAR_STAGING = 1 << 2

PACKER_STATUS_READY = 1 << 0
PACKER_STATUS_PENDING = 1 << 1
PACKER_STATUS_ANY_ERROR = 1 << 2

INST_RE = re.compile(
    r"INST=([0-9a-fA-F]+)\s+ADDR=([0-9a-fA-F]+)\s+BEAT=([0-9a-fA-F]+)\s+DATA=([0-9a-fA-F]+)"
)
IQ_ID_RE = re.compile(r"\bIQ_ID=(\d+)\b")


class StageTimeout(RuntimeError):
    pass


def die(msg, code=1):
    print("[ERR] " + msg, file=sys.stderr)
    sys.exit(code)


def info(msg):
    print("[AIACC_U280] " + msg, flush=True)


def read_hex_lines(path):
    lines = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    return lines


def hex_lines_to_mem_bytes(path):
    out = bytearray()
    for s in read_hex_lines(path):
        if len(s) % 2 != 0:
            die("bad odd-length hex line in {}: {}".format(path, s))
        # load_store_v3 write_hex_file emits byte[15]..byte[0]. Hardware memory
        # byte order is low-address first, so reverse each 16B/line chunk.
        b = bytes.fromhex(s)
        out.extend(b[::-1])
    return bytes(out)


def mem_bytes_to_hex_lines(data):
    if len(data) % 16 != 0:
        die("internal error: result bytes not 16B aligned")
    return [data[i:i + 16][::-1].hex() for i in range(0, len(data), 16)]


class UserBar:
    def __init__(self, dev, size):
        self.dev = dev
        self.size = size
        self.fd = os.open(dev, os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=0)

    def close(self):
        self.mm.close()
        os.close(self.fd)

    def read32(self, off):
        self._check(off, 4)
        return struct.unpack_from("<I", self.mm, off)[0]

    def _barrier(self, off):
        # PCIe BAR MMIO mappings often do not support msync/mmap.flush().
        # A readback is enough to order posted MMIO writes for this runner.
        _ = self.mm[off]

    def write32(self, off, val):
        self._check(off, 4)
        struct.pack_into("<I", self.mm, off, val & 0xFFFFFFFF)
        self._barrier(off)

    def write_beat_strobed(self, off, data_hex, beat_mask):
        self._check(off, 16)
        data = bytes.fromhex(data_hex)
        if len(data) != 16:
            die("IQ DATA is not 16B at 0x{:x}: {}".format(off, data_hex))
        data = data[::-1]
        for i in range(16):
            if (beat_mask >> i) & 1:
                self.mm[off + i] = data[i]
        self._barrier(off)

    def _check(self, off, n):
        if off < 0 or off + n > self.size:
            die("BAR access out of range: off=0x{:x} size={} bar_size=0x{:x}".format(off, n, self.size))


def find_tool(name, tools_dir):
    candidates = []
    if tools_dir:
        candidates.append(Path(tools_dir) / name)
    candidates.extend([
        Path.cwd() / name,
        Path.cwd() / "tools" / name,
        Path.home() / "Jianyuan-host_xdma" / "linux-kernel" / "tools" / name,
        Path.home() / "Jianyuan" / "Jianyuan-host_xdma" / "linux-kernel" / "tools" / name,
        Path("/home/bupt/U280_build/Jianyuan-host_xdma/linux-kernel/tools") / name,
        Path("/home/zwz/Jianyuan-host_xdma/linux-kernel/tools") / name,
        Path("/usr/local/bin") / name,
        Path("/usr/bin") / name,
    ])
    for c in candidates:
        if c.is_file() and os.access(str(c), os.X_OK):
            return str(c)
    searched = "\n  ".join(str(c) for c in candidates)
    die("cannot find executable tool '{}'. Set XDMA_TOOLS_DIR to the directory containing it. Searched:\n  {}".format(name, searched))


def align_up(n, align):
    if align <= 1:
        return n
    return ((n + align - 1) // align) * align


def run_cmd(cmd, dry_run=False):
    info("$ " + " ".join(str(x) for x in cmd))
    if dry_run:
        return
    subprocess.run([str(x) for x in cmd], check=True)


def _dma_chunk(dma_tool, h2c_dev, tmp_dir, name, addr, data, dma_align, dry_run):
    bin_path = tmp_dir / (name + ".bin")
    valid_size = len(data)
    xfer_size = align_up(valid_size, dma_align)
    if xfer_size != valid_size:
        data = data + bytes(xfer_size - valid_size)
    bin_path.write_bytes(data)
    run_cmd([dma_tool, "-d", h2c_dev, "-a", hex(addr), "-s", str(xfer_size), "-f", bin_path], dry_run)
    info("HBM {} <- {} bytes @ 0x{:x} (valid={}, align={})".format(
        name, xfer_size, addr, valid_size, dma_align))


def _load_hbm_segments(case_dir, h2c_dev, tools_dir, stride, tmp_dir, dma_align, dry_run):
    """Slot-rebound layout (slot_map=fpga): per-channel segments at absolute
    window offsets, per package_case_to_fpga.py's segments.json."""
    dma_to_device = "dma_to_device" if dry_run else find_tool("dma_to_device", tools_dir)
    hbm_dir = case_dir / "hbm_data"
    doc = json.loads((hbm_dir / "segments.json").read_text(encoding="utf-8"))
    loaded = 0
    for ch in doc["channels"]:
        smc = int(ch["smc"])
        for seg in ch["segments"]:
            data = hex_lines_to_mem_bytes(hbm_dir / seg["file"])
            if not data:
                continue
            addr = smc * stride + int(seg["offset"])
            _dma_chunk(dma_to_device, h2c_dev, tmp_dir,
                       "ch{}_{}".format(smc, seg["file"].replace(".hex", "")),
                       addr, data, dma_align, dry_run)
            loaded += 1
    if loaded == 0:
        die("segments.json has no non-empty segments")


def load_hbm(case_dir, h2c_dev, tools_dir, stride, tmp_dir, dma_align, dry_run=False):
    if (case_dir / "hbm_data" / "segments.json").is_file():
        return _load_hbm_segments(case_dir, h2c_dev, tools_dir, stride, tmp_dir, dma_align, dry_run)
    dma_to_device = "dma_to_device" if dry_run else find_tool("dma_to_device", tools_dir)
    hbm_dir = case_dir / "hbm_data"
    if not hbm_dir.is_dir():
        die("missing hbm_data dir: {}".format(hbm_dir))
    loaded = 0
    for smc in range(8):
        hex_path = hbm_dir / "memory_init{}.hex".format(smc + 1)
        if not hex_path.is_file():
            continue
        data = hex_lines_to_mem_bytes(hex_path)
        if not data:
            continue
        bin_path = tmp_dir / "memory_init{}_le.bin".format(smc + 1)
        valid_size = len(data)
        xfer_size = align_up(valid_size, dma_align)
        if xfer_size != valid_size:
            data = data + bytes(xfer_size - valid_size)
        bin_path.write_bytes(data)
        addr = smc * stride
        run_cmd([dma_to_device, "-d", h2c_dev, "-a", hex(addr), "-s", str(xfer_size), "-f", bin_path], dry_run)
        info("HBM smc_{} <- {} bytes @ 0x{:x} (valid={}, align={})".format(smc, xfer_size, addr, valid_size, dma_align))
        loaded += 1
    if loaded == 0:
        die("no memory_init*.hex loaded from {}".format(hbm_dir))


def parse_iq_file(path):
    iq_ids = []
    writes = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline()
        m_id = IQ_ID_RE.search(first)
        if m_id:
            iq_ids.append(int(m_id.group(1)))
        for raw in f:
            m = INST_RE.search(raw.strip())
            if not m:
                continue
            _inst, addr_s, beat_s, data_s = m.groups()
            writes.append((int(addr_s, 16), int(beat_s, 16), data_s.zfill(32)))
    return iq_ids, writes


def ordered_stage_iq_files(stage_dir, subq):
    pattern = "iq_inst_*_subq{}.txt".format(subq)
    discovered = list(stage_dir.glob(pattern))
    if not discovered:
        die("no {} under {}".format(pattern, stage_dir))

    data_list = stage_dir / "data_list_inst_subq{}.txt".format(subq)
    if not data_list.is_file():
        files = sorted(discovered)
        info("{} has no {}; fallback IQ order={}".format(
            stage_dir.name, data_list.name, [p.name for p in files]))
        return files

    files = []
    seen = set()
    with open(data_list, "r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                die("bad data_list line {} in {}: {}".format(
                    line_no, data_list, line))
            basename = Path(fields[1]).name
            if not Path(basename).match(pattern):
                continue
            local_path = stage_dir / basename
            if not local_path.is_file():
                die("data_list IQ file is missing from packaged stage: {} -> {}".format(
                    fields[1], local_path))
            if basename in seen:
                die("duplicate IQ file in {}: {}".format(data_list, basename))
            files.append(local_path)
            seen.add(basename)

    discovered_names = {p.name for p in discovered}
    listed_names = {p.name for p in files}
    if discovered_names != listed_names:
        die("{} does not exactly describe stage IQ files; missing={} extra={}".format(
            data_list, sorted(discovered_names - listed_names),
            sorted(listed_names - discovered_names)))
    return files


def beat_mask_span(beat_mask):
    if beat_mask == 0:
        die("bad IQ beat mask: 0")
    lo = None
    hi = None
    for i in range(16):
        if (beat_mask >> i) & 1:
            if lo is None:
                lo = i
            hi = i
    for i in range(lo, hi + 1):
        if ((beat_mask >> i) & 1) == 0:
            die("non-contiguous IQ beat mask is not supported: 0x{:04x}".format(beat_mask))
    return lo, hi


def write_iq_beat_dma(dma_to_device, h2c_dev, tmp_dir, addr, beat_mask, data_hex, dry_run=False):
    lo, hi = beat_mask_span(beat_mask)
    data = bytes.fromhex(data_hex)
    if len(data) != 16:
        die("IQ DATA is not 16B at 0x{:x}: {}".format(addr, data_hex))
    # IQ txt DATA is emitted as byte[15]..byte[0]. AXI byte lane 0 is the low
    # address byte, so reverse before applying the WSTRB-like beat mask.
    data = data[::-1]
    payload = data[lo:hi + 1]
    bin_path = tmp_dir / "iq_{:08x}_{:04x}.bin".format(addr, beat_mask)
    bin_path.write_bytes(payload)
    axi_addr = AXI_LITE_BASE + addr + lo
    run_cmd([dma_to_device, "-d", h2c_dev, "-a", hex(axi_addr), "-s", str(len(payload)), "-f", bin_path], dry_run)


def packer_commit_count(status):
    return (status >> 16) & 0xFFFF


def packer_error_text(status):
    names = []
    for bit, name in (
        (3, "incomplete"),
        (4, "unaligned"),
        (5, "bad_addr"),
        (6, "zero_wstrb"),
    ):
        if status & (1 << bit):
            names.append(name)
    return ",".join(names) if names else "unknown"


def wait_packer_commit(bar, old_count, timeout_s=1.0, poll_s=0.0001):
    expected_count = (old_count + 1) & 0xFFFF
    deadline = time.time() + timeout_s
    status = 0
    while time.time() < deadline:
        status = bar.read32(PACKER_STATUS)
        if status & PACKER_STATUS_ANY_ERROR:
            die("IQ packer error status=0x{:08x} ({})".format(
                status, packer_error_text(status)))
        if (packer_commit_count(status) == expected_count and
                (status & PACKER_STATUS_PENDING) == 0 and
                (status & PACKER_STATUS_READY) != 0):
            return status
        time.sleep(poll_s)
    die("IQ packer commit timeout: old_count={} expected={} status=0x{:08x}".format(
        old_count, expected_count, status))


def write_iq_beat_packer(bar, addr, beat_mask, data_hex, dry_run=False):
    if addr & 0xF:
        die("IQ packer target is not 16B aligned: 0x{:08x}".format(addr))
    if addr < 0x1000 or addr >= 0x5000:
        die("IQ packer target outside IQ window: 0x{:08x}".format(addr))
    if beat_mask == 0 or beat_mask > 0xFFFF:
        die("bad IQ packer WSTRB: 0x{:x}".format(beat_mask))

    raw = bytes.fromhex(data_hex)
    if len(raw) != 16:
        die("IQ DATA is not 16B at 0x{:x}: {}".format(addr, data_hex))
    words = struct.unpack("<4I", raw[::-1])

    if dry_run:
        info("dry-run packer IQ addr=0x{:08x} wstrb=0x{:04x} data={}".format(
            addr, beat_mask, data_hex))
        return

    # Every IQ beat is self-contained. Clear stale staging/error state first,
    # then stage all six fields and explicitly commit one 128-bit beat.
    bar.write32(PACKER_CTRL, PACKER_CTRL_CLEAR_ERROR | PACKER_CTRL_CLEAR_STAGING)
    for off, word in zip(PACKER_DATA, words):
        bar.write32(off, word)
    bar.write32(PACKER_WSTRB, beat_mask)
    bar.write32(PACKER_IQADDR, addr)

    status = bar.read32(PACKER_STATUS)
    if status & PACKER_STATUS_ANY_ERROR:
        die("IQ packer pre-commit error status=0x{:08x} ({})".format(
            status, packer_error_text(status)))
    stage_valid = (status >> 8) & 0x3F
    if stage_valid != 0x3F:
        die("IQ packer staging incomplete before commit: status=0x{:08x} stage_valid=0x{:02x}".format(
            status, stage_valid))

    old_count = packer_commit_count(status)
    bar.write32(PACKER_CTRL, PACKER_CTRL_COMMIT)
    wait_packer_commit(bar, old_count)


def load_stage_iq(bar, stage_dir, subq, tools_dir=None, h2c_dev=None, tmp_dir=None,
                  iq_load_mode="packer", dry_run=False):
    files = ordered_stage_iq_files(stage_dir, subq)
    dma_to_device = None
    if iq_load_mode == "dma":
        if tmp_dir is None or h2c_dev is None:
            die("internal error: dma IQ load requires tmp_dir and h2c_dev")
        dma_to_device = find_tool("dma_to_device", tools_dir)
    elif iq_load_mode not in ("bar", "packer"):
        die("unknown iq_load_mode: {}".format(iq_load_mode))
    iq_mask = 0
    beat_count = 0
    iq_order = []
    for path in files:
        ids, writes = parse_iq_file(path)
        for iq_id in ids:
            if iq_id < 0 or iq_id >= 128:
                die("bad IQ_ID={} in {}".format(iq_id, path))
            iq_mask |= (1 << iq_id)
            iq_order.append(iq_id)
        for addr, beat, data_hex in writes:
            if iq_load_mode == "dma":
                write_iq_beat_dma(dma_to_device, h2c_dev, tmp_dir, addr, beat, data_hex, dry_run=dry_run)
            elif iq_load_mode == "packer":
                write_iq_beat_packer(bar, addr, beat, data_hex, dry_run=dry_run)
            elif not dry_run:
                bar.write_beat_strobed(addr, data_hex, beat)
            beat_count += 1
    info("loaded {}: {} IQ files, {} IQ writes via {}, order={}, IQ_EN=0x{:032x}".format(
        stage_dir.name, len(files), beat_count, iq_load_mode, iq_order, iq_mask))
    return iq_mask


def write_checked_csr(bar, off, value, settle_s, name):
    bar.write32(off, value)
    if settle_s > 0:
        time.sleep(settle_s)
    actual = bar.read32(off)
    if actual != (value & 0xFFFFFFFF):
        die("{} readback mismatch at 0x{:03x}: wrote=0x{:08x} read=0x{:08x}".format(
            name, off, value & 0xFFFFFFFF, actual))


def write_iq_en_and_start(bar, iq_mask, subq, csr_settle_s, dry_run=False):
    words = [(iq_mask >> (32 * i)) & 0xFFFFFFFF for i in range(4)]
    if dry_run:
        info("dry-run start subq={} IQ_EN={}".format(subq, [hex(x) for x in words]))
        return
    write_checked_csr(bar, CSR_CFG, 0x00000001, csr_settle_s, "CFG")
    for index, (off, val) in enumerate(zip(CSR_IQ_EN, words)):
        write_checked_csr(bar, off, val, csr_settle_s, "IQ_EN{}".format(index))
    bar.write32(CSR_START, ((subq & 0x3) << 8) | 0x1)
    if csr_settle_s > 0:
        time.sleep(csr_settle_s)


def ack_interrupt(bar, csr_settle_s):
    int_word = bar.read32(CSR_INT)
    intid_hw = (int_word >> 8) & 0xFF
    int_word = (int_word & ~(0xFF << 16)) | (intid_hw << 16)
    bar.write32(CSR_INT, int_word)
    if csr_settle_s > 0:
        time.sleep(csr_settle_s)
    acked = bar.read32(CSR_INT)
    if acked & 0x1:
        die("interrupt ack did not clear INT: before=0x{:08x} after=0x{:08x}".format(
            int_word, acked))
    return acked


def decode_int_word(int_word):
    return {
        "intr": int_word & 0x1,
        "err": (int_word >> 1) & 0x1,
        "hw": (int_word >> 8) & 0xFF,
        "sw": (int_word >> 16) & 0xFF,
    }


def wait_stage_done(bar, timeout_s, poll_s, csr_settle_s):
    deadline = time.time() + timeout_s
    polls = 0
    last = None
    while time.time() < deadline:
        int_word = bar.read32(CSR_INT)
        done_word = bar.read32(CSR_DONE)
        last = (int_word, done_word)
        if done_word & 0x1:
            if int_word & 0x1:
                ack_interrupt(bar, csr_settle_s)
            return "DONE", int_word, done_word
        polls += 1
        if polls % 100 == 0:
            info("polling... INT=0x{:08x} DONE=0x{:08x}".format(int_word, done_word))
        time.sleep(poll_s)
    int_word, done_word = last if last else (0, 0)
    cfg_word = bar.read32(CSR_CFG)
    iq_words = [bar.read32(off) for off in CSR_IQ_EN]
    packer_status = bar.read32(PACKER_STATUS)
    raise StageTimeout(
        "stage timeout: INT=0x{:08x} DONE=0x{:08x} CFG=0x{:08x} "
        "IQ_EN={} PACKER_STATUS=0x{:08x}".format(
            int_word, done_word, cfg_word,
            ["0x{:08x}".format(word) for word in iq_words], packer_status))


def prepare_device(user_dev, bar_size, iq_load_mode, reset_settle_s,
                   skip_soft_reset=False, dry_run=False):
    if dry_run:
        if skip_soft_reset:
            info("dry-run: SMI soft reset disabled")
        else:
            info("dry-run: SMI soft reset before HBM/IQ load")
        return

    bar = UserBar(user_dev, bar_size)
    try:
        rtl_version = bar.read32(CSR_RTL_VERSION)
        info("RTL_VERSION=0x{:08x}".format(rtl_version))
        if rtl_version != EXPECTED_RTL_VERSION:
            die("unexpected RTL_VERSION 0x{:08x}, expected 0x{:08x}".format(
                rtl_version, EXPECTED_RTL_VERSION))
        if iq_load_mode == "packer":
            packer_cap = bar.read32(PACKER_CAP)
            info("IQ_PACKER_CAP=0x{:08x}".format(packer_cap))
            if packer_cap != PACKER_CAP_IQP1:
                die("unexpected IQ packer CAP 0x{:08x}, expected 0x{:08x}".format(
                    packer_cap, PACKER_CAP_IQP1))

        if skip_soft_reset:
            info("SMI soft reset disabled by --no-soft-reset")
            return

        bar.write32(CSR_CFG, 0x00000002)
        time.sleep(reset_settle_s)
        cfg_word = bar.read32(CSR_CFG)
        int_word = bar.read32(CSR_INT)
        done_word = bar.read32(CSR_DONE)
        if cfg_word != 0 or int_word != 0 or done_word != 0:
            die("SMI soft reset verification failed: CFG=0x{:08x} INT=0x{:08x} DONE=0x{:08x}".format(
                cfg_word, int_word, done_word))
        if iq_load_mode == "packer":
            status = bar.read32(PACKER_STATUS)
            if status != PACKER_STATUS_READY:
                die("IQ packer did not reset cleanly: STATUS=0x{:08x}".format(status))
            info("SMI soft reset PASS: CFG=0 INT=0 DONE=0 PACKER_STATUS=0x{:08x}".format(status))
        else:
            info("SMI soft reset PASS: CFG=0 INT=0 DONE=0")
    finally:
        bar.close()


def run_stages(case_dir, user_dev, bar_size, subq, timeout_s, poll_s,
               tools_dir, h2c_dev, tmp_dir, iq_load_mode, csr_settle_s,
               dry_run=False):
    stages = sorted((case_dir / "smi_iq_inst").glob("stage_*"), key=lambda p: int(p.name.split("_")[-1]))
    if not stages:
        die("missing smi_iq_inst/stage_* under {}".format(case_dir))
    if dry_run:
        bar = None
    else:
        bar = UserBar(user_dev, bar_size)
    try:
        if not dry_run:
            rtl_version = bar.read32(CSR_RTL_VERSION)
            info("RTL_VERSION=0x{:08x}".format(rtl_version))
            if rtl_version != EXPECTED_RTL_VERSION:
                die("unexpected RTL_VERSION 0x{:08x}, expected 0x{:08x}".format(rtl_version, EXPECTED_RTL_VERSION))
            if iq_load_mode == "packer":
                packer_cap = bar.read32(PACKER_CAP)
                info("IQ_PACKER_CAP=0x{:08x}".format(packer_cap))
                if packer_cap != PACKER_CAP_IQP1:
                    die("unexpected IQ packer CAP 0x{:08x}, expected 0x{:08x}".format(
                        packer_cap, PACKER_CAP_IQP1))
                status = bar.read32(PACKER_STATUS)
                info("IQ_PACKER_STATUS=0x{:08x}".format(status))
                if status & PACKER_STATUS_ANY_ERROR:
                    bar.write32(PACKER_CTRL, PACKER_CTRL_CLEAR_ERROR | PACKER_CTRL_CLEAR_STAGING)
        for idx, stage in enumerate(stages):
            info("===== {} begin =====".format(stage.name))
            iq_mask = load_stage_iq(bar, stage, subq, tools_dir=tools_dir, h2c_dev=h2c_dev,
                                    tmp_dir=tmp_dir, iq_load_mode=iq_load_mode, dry_run=dry_run)
            int_before = bar.read32(CSR_INT) if not dry_run else 0
            write_iq_en_and_start(bar, iq_mask, subq, csr_settle_s,
                                  dry_run=dry_run)
            if not dry_run:
                reason, int_word, done_word = wait_stage_done(
                    bar, timeout_s, poll_s, csr_settle_s)
                before = decode_int_word(int_before)
                after = decode_int_word(int_word)
                int_delta = (after["hw"] - before["hw"]) & 0xFF
                info("{} complete by {} INT=0x{:08x} DONE=0x{:08x} "
                     "intid_hw:{}->{} delta={} intid_sw={}".format(
                         stage.name, reason, int_word, done_word,
                         before["hw"], after["hw"], int_delta, after["sw"]))
                if int_delta != 1:
                    info("WARN {} generated {} interrupt events; expected one batch-done event".format(
                        stage.name, int_delta))
    finally:
        if bar is not None:
            bar.close()


def initial_hbm_region(case_dir, smc, byte_offset, size):
    init_path = case_dir / "hbm_data" / "memory_init{}.hex".format(smc + 1)
    if not init_path.is_file():
        return None
    data = hex_lines_to_mem_bytes(init_path)
    if byte_offset + size > len(data):
        return None
    return data[byte_offset:byte_offset + size]


def read_and_compare_stitched(case_dir, dma_from_device, c2h_dev, stride, tmp_dir,
                              dma_align, meta, num_smc, out_bank, out_words,
                              dry_run=False, diagnostic=False):
    """fpga slot-map store (w4f16 D5): every SMC interleaves its 32B slices
    into one contiguous region in a single channel (gr_intlv_addr=0x20):
    block k = 256B, SMC s owns bytes [s*32, s*32+32) of the block.  Read the
    whole region once, then reconstruct each SMC's 2-word-per-block stream
    and compare it against that SMC's golden file."""
    store_byte_base = int(meta["store_byte_base"])
    if out_words % 2:
        die("stitched store expects even out_words, got {}".format(out_words))
    blocks = out_words // 2
    size = blocks * 256
    ch = (store_byte_base >> 28) & 0xFF
    if ch >= num_smc:
        die("stitched store channel {} out of range (num_smc={})".format(ch, num_smc))
    local = store_byte_base & (stride - 1)
    if local + size > stride:
        die("stitched store region [0x{:x}, +0x{:x}) crosses the channel window".format(
            local, size))
    addr = ch * stride + local
    out_bin = tmp_dir / "hbm_out_stitched.bin"
    xfer_size = align_up(size, dma_align)
    try:
        run_cmd([dma_from_device, "-d", c2h_dev, "-a", hex(addr),
                 "-s", str(xfer_size), "-f", out_bin], dry_run)
    except subprocess.CalledProcessError as exc:
        if not diagnostic:
            raise
        info("TIMEOUT DDR DIAG stitched READ_ERROR exit={}".format(exc.returncode))
        return False
    if dry_run:
        info("dry-run: skipped HBM readback compare")
        return
    actual = out_bin.read_bytes()[:size]
    if len(actual) < size:
        die("stitched readback short: got {} bytes, expected {}".format(
            len(actual), size))
    fail_dir = case_dir / "u280_fail_dump"
    pass_count = 0
    fail_count = 0
    for smc in range(num_smc):
        golden_path = case_dir / "smc_{}".format(smc) / "output_bank" / "bank{}_final.hex".format(out_bank)
        if not golden_path.is_file():
            info("smc_{} skip: missing {}".format(smc, golden_path))
            continue
        # SMC s's word stream = bytes [k*256+s*32, +32) of each block k.
        smc_bytes = b"".join(
            actual[k * 256 + smc * 32:k * 256 + smc * 32 + 32]
            for k in range(blocks))
        actual_lines = mem_bytes_to_hex_lines(smc_bytes)
        golden_lines = read_hex_lines(golden_path)[:out_words]
        if actual_lines != golden_lines:
            fail_dir.mkdir(exist_ok=True)
            (fail_dir / "smc{}_actual.hex".format(smc)).write_text(
                "\n".join(actual_lines) + "\n", encoding="utf-8")
            for i, (a, g) in enumerate(zip(actual_lines, golden_lines)):
                if a != g:
                    msg = "smc_{} FAIL line {} actual={} golden={} state=CHANGED_BUT_WRONG dump={}".format(
                        smc, i, a, g, fail_dir)
                    if diagnostic:
                        info("TIMEOUT DDR DIAG " + msg)
                        break
                    die(msg)
            else:
                msg = "smc_{} FAIL length actual={} golden={}".format(
                    smc, len(actual_lines), len(golden_lines))
                if diagnostic:
                    info("TIMEOUT DDR DIAG " + msg)
                else:
                    die(msg)
            fail_count += 1
            continue
        prefix = "TIMEOUT DDR DIAG " if diagnostic else ""
        info("{}smc_{} PASS DDR bit-exact {} lines".format(prefix, smc, out_words))
        pass_count += 1
    if diagnostic:
        info("TIMEOUT DDR DIAG summary: exact={} mismatch={}".format(
            pass_count, fail_count))
        return fail_count == 0 and pass_count > 0
    if pass_count == 0:
        die("no SMC golden compared")
    info("CASE PASS compared_smc={}".format(pass_count))


def read_and_compare(case_dir, c2h_dev, tools_dir, stride, tmp_dir, dma_align,
                     dry_run=False, diagnostic=False):
    dma_from_device = "dma_from_device" if dry_run else find_tool("dma_from_device", tools_dir)
    meta_path = case_dir / "meta_v3.json"
    if not meta_path.is_file():
        die("missing meta_v3.json: {}".format(meta_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    num_smc = int(meta.get("num_smc", 8))
    out_bank = int(meta["out_bank"])
    out_words = int(meta["out_words"])
    word_bytes = int(meta.get("word_bytes", 16))
    store_byte_base = int(meta["store_byte_base"])
    size = out_words * word_bytes
    if word_bytes != 16:
        die("only 16B word_bytes supported, got {}".format(word_bytes))
    if meta.get("slot_map") == "fpga" and int(meta.get("gr_intlv_addr", 0)) == 0x20:
        return read_and_compare_stitched(
            case_dir, dma_from_device, c2h_dev, stride, tmp_dir, dma_align,
            meta, num_smc, out_bank, out_words, dry_run=dry_run,
            diagnostic=diagnostic)

    pass_count = 0
    fail_count = 0
    unchanged_count = 0
    for smc in range(num_smc):
        golden_path = case_dir / "smc_{}".format(smc) / "output_bank" / "bank{}_final.hex".format(out_bank)
        if not golden_path.is_file():
            info("smc_{} skip: missing {}".format(smc, golden_path))
            continue
        out_bin = tmp_dir / "hbm_out_smc{}.bin".format(smc)
        addr = smc * stride + store_byte_base
        xfer_size = align_up(size, dma_align)
        try:
            run_cmd([dma_from_device, "-d", c2h_dev, "-a", hex(addr), "-s", str(xfer_size), "-f", out_bin], dry_run)
        except subprocess.CalledProcessError as exc:
            if not diagnostic:
                raise
            info("TIMEOUT DDR DIAG smc_{} READ_ERROR exit={}".format(smc, exc.returncode))
            fail_count += 1
            continue
        if dry_run:
            continue
        actual = out_bin.read_bytes()[:size]
        actual_lines = mem_bytes_to_hex_lines(actual)
        golden_lines = read_hex_lines(golden_path)[:out_words]
        if actual_lines != golden_lines:
            fail_dir = case_dir / "u280_fail_dump"
            fail_dir.mkdir(exist_ok=True)
            (fail_dir / "smc{}_actual.hex".format(smc)).write_text("\n".join(actual_lines) + "\n", encoding="utf-8")
            initial = initial_hbm_region(case_dir, smc, store_byte_base, size)
            unchanged = initial is not None and actual == initial
            if unchanged:
                unchanged_count += 1
            for i, (a, g) in enumerate(zip(actual_lines, golden_lines)):
                if a != g:
                    msg = "smc_{} FAIL line {} actual={} golden={} state={} dump={}".format(
                        smc, i, a, g,
                        "UNCHANGED_FROM_INIT" if unchanged else "CHANGED_BUT_WRONG",
                        fail_dir)
                    if diagnostic:
                        info("TIMEOUT DDR DIAG " + msg)
                        break
                    die(msg)
            else:
                msg = "smc_{} FAIL length actual={} golden={}".format(
                    smc, len(actual_lines), len(golden_lines))
                if diagnostic:
                    info("TIMEOUT DDR DIAG " + msg)
                else:
                    die(msg)
            fail_count += 1
            continue
        prefix = "TIMEOUT DDR DIAG " if diagnostic else ""
        info("{}smc_{} PASS DDR bit-exact {} lines".format(prefix, smc, out_words))
        pass_count += 1
    if dry_run:
        info("dry-run: skipped HBM readback compare")
        return
    if diagnostic:
        info("TIMEOUT DDR DIAG summary: exact={} mismatch={} unchanged_from_init={}".format(
            pass_count, fail_count, unchanged_count))
        return fail_count == 0 and pass_count > 0
    if pass_count == 0:
        die("no SMC golden compared")
    info("CASE PASS compared_smc={}".format(pass_count))


def main():
    ap = argparse.ArgumentParser(description="Run one AIACC U280 multi_inst_stage case")
    ap.add_argument("--case-dir", required=True, help="multi_inst_stage case directory")
    ap.add_argument("--tools-dir", default=os.environ.get("XDMA_TOOLS_DIR", "/home/bupt/U280_build/Jianyuan-host_xdma/linux-kernel/tools"))
    ap.add_argument("--user-dev", default=os.environ.get("XDMA_USER_DEV", "/dev/xdma0_user"))
    ap.add_argument("--h2c-dev", default=os.environ.get("XDMA_H2C_DEV", "/dev/xdma0_h2c_0"))
    ap.add_argument("--c2h-dev", default=os.environ.get("XDMA_C2H_DEV", "/dev/xdma0_c2h_0"))
    ap.add_argument("--hbm-stride", type=lambda x: int(x, 0), default=DEFAULT_HBM_STRIDE)
    ap.add_argument("--bar-size", type=lambda x: int(x, 0), default=DEFAULT_BAR_SIZE)
    ap.add_argument("--subq", type=int, default=0)
    ap.add_argument("--dma-align", type=lambda x: int(x, 0), default=4096, help="pad DMA transfer sizes to this byte alignment; set 1 to disable")
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--poll-s", type=float, default=0.01)
    ap.add_argument("--csr-settle-s", type=float, default=DEFAULT_CSR_SETTLE_S,
                    help="delay after ordered CSR writes/readbacks")
    ap.add_argument("--reset-settle-s", type=float, default=1.0,
                    help="delay after SMI soft reset before verification")
    ap.add_argument("--no-soft-reset", action="store_true",
                    help="do not reset SMI before loading the case (diagnosis only)")
    ap.add_argument("--iq-load-mode", choices=["packer", "dma", "bar"],
                    default=os.environ.get("AIACC_IQ_LOAD_MODE", "packer"),
                    help="load IQ through the IQP1 4x32-bit packer (default), SGDMA AXI-Lite, or legacy BAR mmap")
    ap.add_argument("--dry-run", action="store_true", help="validate files and print actions without touching devices")
    args = ap.parse_args()

    case_dir = Path(args.case_dir).resolve()
    if not case_dir.is_dir():
        die("case dir does not exist: {}".format(case_dir))

    with tempfile.TemporaryDirectory(prefix="aiacc_u280_runner_") as td:
        tmp_dir = Path(td)
        prepare_device(args.user_dev, args.bar_size, args.iq_load_mode,
                       args.reset_settle_s,
                       skip_soft_reset=args.no_soft_reset,
                       dry_run=args.dry_run)
        load_hbm(case_dir, args.h2c_dev, args.tools_dir, args.hbm_stride, tmp_dir, args.dma_align, dry_run=args.dry_run)
        try:
            run_stages(case_dir, args.user_dev, args.bar_size, args.subq, args.timeout_s, args.poll_s,
                       args.tools_dir, args.h2c_dev, tmp_dir, args.iq_load_mode,
                       args.csr_settle_s, dry_run=args.dry_run)
        except StageTimeout as exc:
            info("ERROR " + str(exc))
            info("stage timeout: reading output DDR before resetting or exiting")
            read_and_compare(case_dir, args.c2h_dev, args.tools_dir, args.hbm_stride,
                             tmp_dir, args.dma_align, dry_run=False, diagnostic=True)
            die(str(exc))
        read_and_compare(case_dir, args.c2h_dev, args.tools_dir, args.hbm_stride, tmp_dir, args.dma_align, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
