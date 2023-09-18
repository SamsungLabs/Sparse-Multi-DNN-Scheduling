//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Design Name: 
// Module Name: Calculate Linear Rate
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: Calculate Linear Rate: (num_nonzeros/num_out) * (1/avg_sparsity),  
// 								avg_sparsity means the avarage ratio of non-zeros
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module Calc_Linear_Rate
	#(
		parameter SCORE_BITWIDTH = 32,
		parameter DATA_WIDTH = 16
    )
	(
		input clk,
		input reset,

		input [SCORE_BITWIDTH-1:0] avg_sparsity_dat,
		input avg_sparsity_vld,
		input [16-1:0] shape_dat,
		input shape_vld,
		input [16-1:0] num_nonzeros,

		output [SCORE_BITWIDTH-1:0] linear_rate_dat_out,
		output linear_rate_vld_out
	);

	wire [SCORE_BITWIDTH-1:0]  measured_sparsity_dat;
	wire           measured_sparsity_vld;

	// Divider here
	sif_div 
	# (
			.WIDTH(SCORE_BITWIDTH)
		) u_sif_div( // 16 bits input, SCORE_BITWIDTH bits outputs divider
			.clk(clk),
			.A_vld(1'b1),
			.A_dat(num_nonzeros),
			.A_rdy(),
			.B_vld(shape_vld),
			.B_dat(shape_dat),
			.B_rdy(),
			.P_vld(measured_sparsity_vld),
			.P_dat(measured_sparsity_dat),
			.P_rdy(1'b1)
	);

	// Multipiler here
	sif_mult 
	# (
			.WIDTH(SCORE_BITWIDTH)
		) u_sif_mult( // 32 bits input, SCORE_BITWIDTH bits outputs multipiler
			.clk(clk),
			.A_vld(measured_sparsity_vld),
			.A_dat(measured_sparsity_dat),
			.A_rdy(),
			.B_vld(avg_sparsity_vld),
			.B_dat(avg_sparsity_dat),
			.B_rdy(),
			.P_vld(linear_rate_vld_out),
			.P_dat(linear_rate_dat_out),
			.P_rdy(1'b1)
	);


endmodule