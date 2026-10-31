# Vitis HLS batch script for the vadd hello-world (PYNQ-Z2, XC7Z020).
# UNTESTED: written before any AMD tool was installed. Commands are the long-standing Tcl API; check that
# your 2025.x install still accepts them.
#
# run from this directory, in a shell where the Vitis settings script has been sourced:
#   vitis-run --mode hls --tcl run_hls.tcl
# If your install does not have vitis-run --tcl, try:  vitis_hls -f run_hls.tcl
#
# Result: ./vadd_prj/solution1/impl/ip  (Vivado IP-catalog repo used by ../../vivado/vadd_pynqz2/build.tcl)

open_project -reset vadd_prj
set_top vadd
add_files vadd.cpp
add_files -tb vadd_tb.cpp
open_solution -reset solution1 -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default      ;# 100 MHz

csim_design                                 ;# C simulation: testbench must print TEST PASSED
csynth_design                               ;# C synthesis: read the latency/resource report afterwards
cosim_design -rtl verilog                   ;# C/RTL co-simulation: must also pass
export_design -format ip_catalog -output ./vadd_ip.zip
exit
