# Vivado batch script: Zynq PS + vadd IP -> bitstream + hwh for PYNQ-Z2.
# UNTESTED: written before any AMD tool was installed. Expect to fix small things on first run.
#
# Prerequisites
#   1. ../../hls/vadd/run_hls.tcl has produced ../../hls/vadd/vadd_ip.zip
#   2. PYNQ-Z2 board files are installed in Vivado (from the TUL / PYNQ board files package), so the part
#      "tul.com.tw:pynq-z2:part0:1.0" exists. Check with:  get_board_parts *pynq*
#
# run (shell with Vivado settings sourced):
#   vivado -mode batch -source build.tcl
# outputs: ./out/vadd.bit and ./out/vadd.hwh   (PYNQ needs both, with the SAME base name)

set proj_name vadd_pynqz2
set ip_zip    [file normalize ../../hls/vadd/vadd_ip.zip]
set out_dir   [file normalize ./out]
file mkdir $out_dir

create_project -force $proj_name ./$proj_name -part xc7z020clg400-1
set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project]

# unpack the HLS IP into a repo and register it
set repo [file normalize ./ip_repo]
file delete -force $repo
file mkdir $repo
set_property ip_repo_paths $repo [current_project]
update_ip_catalog -add_ip $ip_zip -repo_path $repo

create_bd_design design_1
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7 processing_system7_0]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1"} $ps
set_property -dict [list CONFIG.PCW_USE_S_AXI_HP0 {1}] $ps         ;# AXI master port for the HLS IP to reach DDR

set vadd [create_bd_cell -type ip -vlnv xilinx.com:hls:vadd vadd_0]

# PS (master GP0) -> vadd control; vadd m_axi -> PS HP0 -> DDR. Automation inserts interconnects + resets.
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master "/processing_system7_0/M_AXI_GP0" Clk "Auto"} [get_bd_intf_pins vadd_0/s_axi_control]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Slave "/processing_system7_0/S_AXI_HP0" Clk "Auto"} [get_bd_intf_pins vadd_0/m_axi_gmem]

assign_bd_address
validate_bd_design
save_bd_design

make_wrapper -files [get_files design_1.bd] -top
add_files -norecurse [file normalize ./$proj_name/$proj_name.gen/sources_1/bd/design_1/hdl/design_1_wrapper.v]
set_property top design_1_wrapper [current_fileset]

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

# report timing status and copy artifacts under one base name
open_run impl_1
report_timing_summary -file $out_dir/timing_summary.rpt
report_utilization    -file $out_dir/utilization.rpt
file copy -force ./$proj_name/$proj_name.runs/impl_1/design_1_wrapper.bit $out_dir/vadd.bit
file copy -force ./$proj_name/$proj_name.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh $out_dir/vadd.hwh
puts "DONE: $out_dir/vadd.bit and vadd.hwh. Check timing_summary.rpt: WNS must be >= 0."
exit
