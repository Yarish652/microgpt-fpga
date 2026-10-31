"""
Run on the PYNQ-Z2 (Jupyter terminal or notebook cell), with vadd.bit and vadd.hwh in the same folder.
UNTESTED: written before any board work. Register names depend on the HLS version, so this prints
the register map first; adjust REG_* below if your names differ.

    sudo python3 vadd_pynq.py
"""
import numpy as np
from pynq import Overlay, allocate

N = 1024
ol = Overlay("vadd.bit")                      # loads vadd.hwh next to it
ip = ol.vadd_0
print(ip.register_map)                        # read this once: names of the pointer registers

a = allocate(shape=(N,), dtype=np.int32)
b = allocate(shape=(N,), dtype=np.int32)
c = allocate(shape=(N,), dtype=np.int32)
rng = np.random.default_rng(0)
a[:] = rng.integers(-10_000, 10_000, N, dtype=np.int32)
b[:] = rng.integers(-10_000, 10_000, N, dtype=np.int32)
c[:] = 0
for buf in (a, b, c):
    buf.flush()

regs = ip.register_map


def set_ptr(name, buf):
    """Pointer args appear as <name>_1 (low 32 bits) and, if present, <name>_2 (high 32 bits)."""
    setattr(regs, name + "_1", buf.physical_address & 0xFFFFFFFF)
    if hasattr(regs, name + "_2"):
        setattr(regs, name + "_2", buf.physical_address >> 32)


set_ptr("a", a)
set_ptr("b", b)
set_ptr("c", c)
regs.n = N
ip.write(0x00, 1)                              # ap_start
while (ip.read(0x00) & 0x2) == 0:              # ap_done
    pass

c.invalidate()
expected = a + b
ok = np.array_equal(c, expected)
print("vadd on board:", "PASS" if ok else "FAIL")
if not ok:
    bad = np.flatnonzero(c != expected)
    print("first mismatches:", bad[:10], c[bad[:10]], expected[bad[:10]])
