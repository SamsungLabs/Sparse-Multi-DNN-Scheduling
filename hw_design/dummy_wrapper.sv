//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Design Name: 
// Module Name: Dummy_Wrapper
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: A wrapper to wrap dummy NPU and scheduler
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module Dummy_Wrapper
	#(
		parameter DATA_BITWIDTH = 16,
		parameter ADDR_BITWIDTH = 10,
		
		parameter DATA_WIDTH = 16,
		parameter ADDR_WIDTH = 9,
		
		// GLB Cluster parameters. This TestBench uses only 1 of each
		parameter NUM_GLB_IACT = 1,
		parameter NUM_GLB_PSUM = 1,
		parameter NUM_GLB_WGHT = 1,
		
		parameter ADDR_BITWIDTH_GLB = 10,
		parameter ADDR_BITWIDTH_SPAD = 9,
		
		parameter NUM_ROUTER_PSUM = 1,
		parameter NUM_ROUTER_IACT = 1,
		parameter NUM_ROUTER_WGHT = 1,
				
		parameter int kernel_size = 3,
		parameter int act_size = 5,
		
		parameter int X_dim = 3,
		parameter int Y_dim = 3,
		
		parameter W_READ_ADDR = 0, 
		parameter A_READ_ADDR = 0,
		
		parameter W_LOAD_ADDR = 0,  
		parameter A_LOAD_ADDR = 0,
		
		parameter PSUM_READ_ADDR = 0,
		parameter PSUM_LOAD_ADDR = 0,

		// For Dysta_Scheduler
    parameter REQST_DEPTH = 9, // means 2^8 number of requests maximally
    parameter COMPUTE_OPT = 0,
    parameter SCORE_BITWIDTH = 16 // 16 (FP16) or 32 (FP32)
    )
	(
		input clk,
		input reset,
		//	Ports for dummy NPU, follow EyerissV2 interface  //
		//		PE Cluster Interface
		input start,
		output load_done,
		
		input load_en_wght,
		input load_en_act,
		
    output [DATA_WIDTH-1:0] pe_out[X_dim-1:0],
		output compute_done,

		//		GLB Cluster Interface
		input write_en_iact,
		input write_en_wght,
		
		input [DATA_WIDTH-1:0] w_data_iact,
		input [ADDR_WIDTH-1:0] w_addr_iact,
		
		input [DATA_WIDTH-1:0] w_data_wght,
		input [ADDR_WIDTH-1:0] w_addr_wght,
		
		input [ADDR_WIDTH-1:0] w_addr_psum,		
				
		output [DATA_WIDTH-1:0] r_data_psum,
		input [ADDR_WIDTH-1:0] r_addr_psum,
	
		input read_req_iact,
		input read_req_psum,
		input read_req_wght,
		input [ADDR_WIDTH-1:0] r_addr_iact,
		input [ADDR_WIDTH-1:0] r_addr_wght,

		//		WGHT Router Ports
		input [3:0] router_mode_wght,
		input [3:0] router_mode_iact,
		input [3:0] router_mode_psum,


		//	Ports for scheduler  //
		output scheduler_rdy,
		input [0:2] request_control,
		input [0:31] request_data

	);


  wire [DATA_WIDTH-1:0] pe_out_npu[X_dim-1:0];
	wire pe_vld_npu;
	wire compute_done_npu;
	wire last_layer_done_npu;

	wire [REQST_DEPTH-1:0] sel_task;
	wire start_comp_npu;

	Dysta_Scheduler
	# (
		.X_dim(X_dim),
    .REQST_DEPTH(REQST_DEPTH), // means 2^8 number of requests maximally
		.COMPUTE_OPT(COMPUTE_OPT),
    .SCORE_BITWIDTH(SCORE_BITWIDTH) // 16 (FP16) or 32 (FP32)
	) u_Dysta_Scheduler
	(
		.clk(clk),
		.reset(reset),
		.in_dat(pe_out_npu),
		.out_dat(pe_out),
		.in_vld(pe_vld_npu),
		.compute_done_npu(compute_done_npu),
		.last_layer_done_npu(last_layer_done_npu),
    .sel_task(sel_task),
    .start_comp_npu(start_comp_npu),
		.scheduler_rdy(scheduler_rdy),
		.request_control(request_control),
		.request_data(request_data)
	);

	Dummy_NPU
	# (
		.DATA_BITWIDTH(DATA_BITWIDTH),
		.ADDR_BITWIDTH(ADDR_BITWIDTH),
		
		.DATA_WIDTH(DATA_WIDTH),
		.ADDR_WIDTH(ADDR_WIDTH),
		
		// GLB Cluster parameters. This TestBench uses only 1 of each
		.NUM_GLB_IACT(NUM_GLB_IACT),
		.NUM_GLB_PSUM(NUM_GLB_PSUM),
		.NUM_GLB_WGHT(NUM_GLB_WGHT),
		
		.ADDR_BITWIDTH_GLB(ADDR_BITWIDTH_GLB),
		.ADDR_BITWIDTH_SPAD(ADDR_BITWIDTH_SPAD),
		
		.NUM_ROUTER_PSUM(NUM_ROUTER_PSUM),
		.NUM_ROUTER_IACT(NUM_ROUTER_IACT),
		.NUM_ROUTER_WGHT(NUM_ROUTER_WGHT),
				
		.kernel_size(kernel_size),
		.act_size(act_size),
		
		.X_dim(X_dim),
		.Y_dim(Y_dim),
		
		.W_READ_ADDR(W_READ_ADDR), 
		.A_READ_ADDR(A_READ_ADDR),
		
		.W_LOAD_ADDR(W_LOAD_ADDR),  
		.A_LOAD_ADDR(A_LOAD_ADDR),
		
		.PSUM_READ_ADDR(PSUM_READ_ADDR),
		.PSUM_LOAD_ADDR(PSUM_LOAD_ADDR),

		// Control from scheduler
    .REQST_DEPTH(REQST_DEPTH) // means 2^8 number of requests maximally
	) u_Dummy_NPU
	(

		.clk(clk),
		.reset(reset),
		
		//PE Cluster Interface
		.start(start),
		.load_done(load_done),
		
		.load_en_wght(load_en_wght),
		.load_en_act(load_en_act),
		
    .pe_out(pe_out_npu),
		.pe_vld(pe_vld_npu),
		.compute_done(compute_done_npu),
		.last_layer_done(last_layer_done_npu),
    .sel_task(sel_task),
    .start_comp_npu(start_comp_npu),

		//GLB Cluster Interface

		.write_en_iact(write_en_iact),
		.write_en_wght(write_en_wght),
		
		.w_data_iact(w_data_iact),
		.w_addr_iact(w_addr_iact),
		
		.w_data_wght(w_data_wght),
		.w_addr_wght(w_addr_wght),
		
		.w_addr_psum(w_addr_psum),		
				
		.r_data_psum(r_data_psum),
		.r_addr_psum(r_addr_psum),
	
		.read_req_iact(read_req_iact),
		.read_req_psum(read_req_psum),
		.read_req_wght(read_req_wght),
		
		.r_addr_iact(r_addr_iact),
		.r_addr_wght(r_addr_wght),

		
		//WGHT Router Ports
		.router_mode_wght(router_mode_wght),
		.router_mode_iact(router_mode_iact),
		.router_mode_psum(router_mode_psum)
	);

endmodule