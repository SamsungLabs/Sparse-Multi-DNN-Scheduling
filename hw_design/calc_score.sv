//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Design Name: 
// Module Name: Calculate Score
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: Compute score, linear_rate * avg_latency + BETA * (ddl - sys_clk) 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module Calc_Score
	#(
		parameter SCORE_BITWIDTH = 32,
		parameter DATA_WIDTH = 16
    )
	(
		input clk,
		input reset,

		input [SCORE_BITWIDTH-1:0] avg_lat_dat,
		input avg_lat_vld,
    input [SCORE_BITWIDTH-1:0] linear_rate_dat_in,


		input [16-1:0] exe_clk,
		input [16-1:0] norm_isolation_dat,
		input norm_isolation_vld,

		input [SCORE_BITWIDTH-1:0] ddl,
    input [SCORE_BITWIDTH-1:0] sys_clk,

    output [SCORE_BITWIDTH-1:0] score_dat,
    output score_vld
	);

	localparam BETA = 0.01;

	// sys time - last execute time

	wire [SCORE_BITWIDTH-1:0]	wait_dat;
	wire         	wait_vld;

	fixed_sub 
	# (
			.WIDTH(SCORE_BITWIDTH)
		) u_sif_sub_wait ( // SCORE_BITWIDTH bits input, SCORE_BITWIDTH bits outputs multipiler
			.clk(clk),
			.A_vld(1'b1),
			.A_dat(sys_clk),
			.A_rdy(),
			.B_vld(1'b1),
			.B_dat(exe_clk),
			.B_rdy(),
			.S_vld(wait_vld),
			.S_dat(wait_dat),
			.S_rdy(1'b1)
	);

	// norm_isolation * wait time

	wire [SCORE_BITWIDTH-1:0]	penalty_dat;
	wire         	penalty_vld;

	sif_mult 
	# (
			.WIDTH(SCORE_BITWIDTH),
			.FIXED_POINT(1)
		) u_sif_mult_penalty ( // SCORE_BITWIDTH bits input, SCORE_BITWIDTH bits outputs multipiler
			.clk(clk),
			.A_vld(wait_vld),
			.A_dat(wait_dat),
			.A_rdy(1'b1),
			.B_vld(norm_isolation_vld),
			.B_dat(norm_isolation_dat),
			.B_rdy(1'b1),
			.P_vld(penalty_vld),
			.P_dat(penalty_dat),
			.P_rdy(1'b1)
	);


  // deadline - sys time
	wire [SCORE_BITWIDTH-1:0]	slack_dat;
	wire         	slack_vld;

	fixed_sub 
	# (
			.WIDTH(SCORE_BITWIDTH)
		) u_sif_sub_slack ( // SCORE_BITWIDTH bits input, SCORE_BITWIDTH bits outputs multipiler
			.clk(clk),
			.A_vld(1'b1),
			.A_dat(ddl),
			.A_rdy(),
			.B_vld(1'b1),
			.B_dat(sys_clk),
			.B_rdy(),
			.S_vld(slack_vld),
			.S_dat(slack_dat),
			.S_rdy(1'b1)
	);

	reg [SCORE_BITWIDTH-1:0]	clamp_slack_dat_r;
	reg         	clamp_slack_vld_r;

	// Clamp to 0~1
	always @(posedge clk or posedge reset)
	if(reset) begin
		clamp_slack_dat_r <= 0;
		clamp_slack_vld_r <= 0;
	end
	else begin
		clamp_slack_vld_r <= slack_vld;
		if (slack_dat < 0) begin
			clamp_slack_dat_r <= 0;
		end
		else begin
			clamp_slack_dat_r <= slack_dat;
		end
	end

	// slack + penalty

	wire [SCORE_BITWIDTH-1:0]	cost_dat;
	wire         	cost_vld;

	fixed_add
	# (
			.WIDTH(SCORE_BITWIDTH)
		) u_sif_add_cost ( // SCORE_BITWIDTH bits input, SCORE_BITWIDTH bits outputs multipiler
			.clk(clk),
			.A_vld(clamp_slack_vld_r),
			.A_dat(clamp_slack_dat_r),
			.A_rdy(),
			.B_vld(penalty_vld),
			.B_dat(penalty_dat),
			.B_rdy(),
			.S_vld(cost_vld),
			.S_dat(cost_dat),
			.S_rdy(1'b1)
	);

	// convert fixed to fp

	wire [SCORE_BITWIDTH-1:0]	fp_cost_dat;
	wire         	fp_cost_vld;

	fixed_to_fp
	# (
			.WIDTH(SCORE_BITWIDTH)
		) u_fixed_to_fp ( // SCORE_BITWIDTH bits input, SCORE_BITWIDTH bits outputs multipiler
			.clk(clk),
			.A_vld(cost_vld),
			.A_dat(cost_dat),
			.A_rdy(),
			.P_vld(fp_cost_vld),
			.P_dat(fp_cost_dat),
			.P_rdy(1'b1)
	);


	// Constant Multipiler
	wire [SCORE_BITWIDTH-1:0]	estimated_lat_dat;
	wire         	estimated_lat_vld;

	sif_mult 
	# (
			.WIDTH(SCORE_BITWIDTH)
		) u_sif_mult_beta( // SCORE_BITWIDTH bits input, SCORE_BITWIDTH bits outputs multipiler
			.clk(clk),
			.A_vld(1'b1),
			.A_dat(BETA),
			.A_rdy(),
			.B_vld(fp_cost_vld),
			.B_dat(fp_cost_dat),
			.B_rdy(),
			.P_vld(beta_slack_vld),
			.P_dat(beta_slack_dat),
			.P_rdy(1'b1)
	);


	// linear_rate * avg_latency
	wire [SCORE_BITWIDTH-1:0]	estimated_lat_dat;
	wire         	estimated_lat_vld;

	sif_mult 
	# (
			.WIDTH(SCORE_BITWIDTH)
		) u_sif_mult_score( // SCORE_BITWIDTH bits input, SCORE_BITWIDTH bits outputs multipiler
			.clk(clk),
			.A_vld(1'b1),
			.A_dat(linear_rate_dat),
			.A_rdy(),
			.B_vld(avg_lat_vld),
			.B_dat(avg_lat_dat),
			.B_rdy(),
			.P_vld(estimated_lat_vld),
			.P_dat(estimated_lat_dat),
			.P_rdy(1'b1)
	);

  // linear_rate * avg_latency + BETA * (ddl - sys_clk)
	sif_addsub 
	# (
			.WIDTH(SCORE_BITWIDTH)
		) u_sif_addsub_score( // SCORE_BITWIDTH bits input, SCORE_BITWIDTH bits outputs multipiler
			.clk(clk),
			.is_sub(1'b0),
			.A_vld(1'b1),
			.A_dat(beta_slack_dat),
			.A_rdy(),
			.B_vld(estimated_lat_vld),
			.B_dat(estimated_lat_dat),
			.B_rdy(),
			.S_vld(score_vld),
			.S_dat(score_dat),
			.S_rdy(1'b1)
	);


endmodule