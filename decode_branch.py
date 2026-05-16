#!/usr/bin/env python3
import struct

# Read the first 4 bytes of space-nx.nro
with open("space-nx-master/spacenx.nro", "rb") as f:
    first_insn = struct.unpack("<I", f.read(4))[0]

print(f"First instruction (little-endian): 0x{first_insn:08x}")

# ARM64 B instruction format: bits [25-0] = signed offset
# The offset is in units of 4 bytes
if (first_insn & 0xFC000000) == 0x14000000:
    offset_units = first_insn & 0x03FFFFFF
    # Sign extend if negative
    if offset_units & 0x02000000:
        offset_units -= 0x04000000
    offset_bytes = offset_units * 4
    print(f"This is a B (branch) instruction with offset {offset_units} units ({offset_bytes} bytes)")
    print(f"Branch target: PC + {offset_bytes:#x} = 0x{offset_bytes:#x}")
else:
    print(f"Not a B instruction: opcode = {(first_insn >> 26) & 0x3F:#x}")
