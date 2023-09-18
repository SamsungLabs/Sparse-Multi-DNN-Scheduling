//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Design Name: 
// Module Name: Dummy_NPU
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: A dummy NPU, with some single logics to avoid that ports are 
// 								optimized away during synthesis&place&route
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module Dummy_NPU
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
		// For scheduler
    parameter REQST_DEPTH = 8 // means 2^8 number of requests maximally

    )
	(
		input clk,
		input reset,
		
		//PE Cluster Interface
		input start,
		output load_done,
		
		input load_en_wght,
		input load_en_act,
		
    output [DATA_WIDTH-1:0] pe_out[X_dim-1:0],
		output pe_vld, // indiate the pe_outis vld, we add this for monitoring the sparsity
		output compute_done,
		output last_layer_done,
		
    // Get from scheduler
    input [REQST_DEPTH-1:0] sel_task,
    input start_comp_npu,

		//GLB Cluster Interface

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

		
		//WGHT Router Ports
		input [3:0] router_mode_wght,
		input [3:0] router_mode_iact,
		input [3:0] router_mode_psum
	);

  // Some simple and logic to make sure the input/output ports are not optmized away
  // Apply and reduction operation for router mode related signals
  reg router_mode_reduce;
  always @(posedge clk or posedge reset)
  if(reset) begin
    router_mode_reduce <= 0;
  end
  else begin
    router_mode_reduce <= (&router_mode_wght) || (&router_mode_iact) || (&router_mode_psum);
  end

  // Apply and reduction operation for address related signals
  reg addr_reduce;
  always @(posedge clk or posedge reset)
  if(reset) begin
    addr_reduce <= 0;
  end
  else begin
    addr_reduce <= (&w_addr_iact) || (&w_addr_wght) || (&w_addr_psum) || (&r_addr_psum) || (&r_addr_iact) || (&r_addr_wght);
  end
  
  // Generate load_done signal
  reg load_done_r;
  always @(posedge clk or posedge reset)
  if(reset) begin
    load_done_r <= 0;
  end
  else begin
    load_done_r <= load_en_wght && load_en_act && read_req_iact && read_req_psum;
  end
  assign load_done = load_done_r;

  // Generate compute_done signal
  reg compute_done_r;
  always @(posedge clk or posedge reset)
  if(reset) begin
    compute_done_r <= 0;
  end
  else begin
    compute_done_r <= write_en_iact && write_en_wght || addr_reduce || (&sel_task);
  end
  assign compute_done = compute_done_r;

  // Generate PE out, use data_iact. 
  generate
  genvar i;
  for(i=0; i<X_dim; i++) 
    begin:gen_X
      assign pe_out[i] = w_data_iact;
    end
  endgenerate

  // Generate PSUM DATA, use data_weight. 
  assign r_data_psum = w_data_wght;

  // Generate last_layer_done signal
  reg last_layer_done_r;
  always @(posedge clk or posedge reset)
  if(reset) begin
    last_layer_done_r <= 0;
  end
  else begin
    last_layer_done_r <= read_req_wght || router_mode_reduce && start_comp_npu;
  end
  assign last_layer_done = last_layer_done_r;

endmodule