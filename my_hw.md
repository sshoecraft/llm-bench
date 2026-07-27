● Hardware Summary — clyde
 
  Motherboard
  - ASUS Z10PE-D8 WS (dual-socket LGA 2011-v3 workstation board, Rev 1.xx)
  - BIOS: AMI 4301, 06/05/2020
  - Board generation: ~2015 (C612 chipset era); ~10–11 years old
 
  CPU — dual socket
  - 2× Intel Xeon E5-2680 v4 (Broadwell-EP, 14C/28T @ 2.4 GHz base, 3.3 GHz turbo)
  - Total: 28 cores / 56 threads, 70 MiB L3 (35 MiB per socket), 2 NUMA nodes
  - Launched Q1 2016; ~10 years old
  Memory
  - 96 GB DDR4 ECC RDIMM (6× 16 GB Samsung, 2 of 8 DIMM slots empty: DIMM_C1, DIMM_G1)
  - Modules are mixed 2400 and 2666 MT/s parts, all running at 2400 MT/s (limited by Broadwell-EP IMC and/or mixed-speed downclock)
  - ECC active (Multi-bit ECC)
 
  GPUs
  - 4× NVIDIA GeForce RTX 3090 (GA102, 24 GB GDDR6X each = 96 GB VRAM total)
  - Driver 595.58.03, mixed VBIOS revisions (likely different AIB vendors)
  - 3090 launched Sept 2020; ~5–6 years old
 
  System
  - Ubuntu, kernel 6.8.0-101 (Feb 2026)
  - Uptime 13.6 h, load avg ~3, 32 GiB RAM in use, 3.9 GiB swap used
 
  Bottom line: ~2016-era dual-Xeon Broadwell workstation (28C/56T, 96 GB DDR4-2400 ECC) retrofitted with a 4× RTX 3090 GPU stack — heavy LLM/inference rig built on aging server silicon.
