//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Design Name: 
// Module Name: Unified Calculate Unit for both score and linear rate
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: Two modes controlled by mode signal:
//              1. mode == 0: Calculate Linear Rate, (num_nonzeros/num_out) * (1/avg_sparsity),  
// 								avg_sparsity means the avarage ratio of non-zeros
//              2. mode == 1: Compute score, linear_rate * avg_latency + BETA * (ddl - sys_clk) 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module Calc_Unit
	#(
		parameter SCORE_BITWIDTH = 32,
		parameter DATA_WIDTH = 16
    )
	(
		input clk,
		input reset,

		input mode,

		// Calcuate Linear Rate
		input [SCORE_BITWIDTH-1:0] avg_sparsity_dat,
		input avg_sparsity_vld,
		input [16-1:0] shape_dat,
		input shape_vld,
		input [16-1:0] num_nonzeros,
		input [16-1:0] exe_clk,
		input [16-1:0] norm_isolation_dat,
		input norm_isolation_vld,

		output [SCORE_BITWIDTH-1:0] linear_rate_dat_out,
		output linear_rate_vld_out,
		
		// Calcuate Score
		input [SCORE_BITWIDTH-1:0] avg_lat_dat,
		input avg_lat_vld,
    input [SCORE_BITWIDTH-1:0] linear_rate_dat_in,

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

  // DeMUX
  wire mult_beta_sparsity_A_vld;
  wire mult_beta_sparsity_B_vld;
  wire mult_beta_sparsity_P_vld;

  wire [SCORE_BITWIDTH-1:0] mult_beta_sparsity_A_dat;
  wire [SCORE_BITWIDTH-1:0] mult_beta_sparsity_B_dat;
  wire [SCORE_BITWIDTH-1:0] mult_beta_sparsity_P_dat;

  assign mult_beta_sparsity_A_vld = mode? 1'b1 : shape_vld;
  assign mult_beta_sparsity_B_vld = mode? fp_cost_vld : 1'b1;

  assign mult_beta_sparsity_A_dat = mode? BETA : shape_dat;
  assign mult_beta_sparsity_B_dat = mode? fp_cost_dat : num_nonzeros;

	sif_mult 
	# (
			.WIDTH(SCORE_BITWIDTH)
		) u_sif_mult_beta_sparsity ( // SCORE_BITWIDTH bits input, SCORE_BITWIDTH bits outputs multipiler, use to 1. beta multiply 2. get sparsity
			.clk(clk),
			.A_vld(mult_beta_sparsity_A_vld),
			.A_dat(mult_beta_sparsity_A_dat),
			.A_rdy(),
			.B_vld(mult_beta_sparsity_B_vld),
			.B_dat(mult_beta_sparsity_B_dat),
			.B_rdy(),
			.P_vld(mult_beta_sparsity_P_vld),
			.P_dat(mult_beta_sparsity_P_dat),
			.P_rdy(1'b1)
	);

  // DeMUX
  reg mult_beta_P_vld;
  reg mult_sparsity_P_vld;

  reg [SCORE_BITWIDTH-1:0] mult_beta_P_dat;
  reg [SCORE_BITWIDTH-1:0] mult_sparsity_P_dat;

  always @ ( * )
  begin
      case(mode)
      1'b0 :  begin
                mult_sparsity_P_dat = mult_beta_sparsity_P_dat;
                mult_sparsity_P_vld = mult_beta_sparsity_P_vld;
              end
      1'b1 :  begin
                mult_beta_P_dat = mult_beta_sparsity_P_dat;
                mult_beta_P_vld = mult_beta_sparsity_P_vld;
              end
      endcase
  end


	// linear_rate * avg_latency

  wire mult_score_linear_A_vld;
  wire mult_score_linear_B_vld;
  wire mult_score_linear_P_vld;

  wire [SCORE_BITWIDTH-1:0] mult_score_linear_A_dat;
  wire [SCORE_BITWIDTH-1:0] mult_score_linear_B_dat;
  wire [SCORE_BITWIDTH-1:0] mult_score_linear_P_dat;

  assign mult_score_linear_A_vld = mode? avg_lat_vld : avg_sparsity_vld;
  assign mult_score_linear_B_vld = mode? 1'b1 : mult_sparsity_P_vld;

  assign mult_score_linear_A_dat = mode? avg_lat_dat : avg_sparsity_dat;
  assign mult_score_linear_B_dat = mode? linear_rate_dat_in : mult_sparsity_P_dat;


	sif_mult 
	# (
			.WIDTH(SCORE_BITWIDTH)
		) u_sif_mult_score_linear ( // SCORE_BITWIDTH bits input, SCORE_BITWIDTH bits outputs multipiler
			.clk(clk),
			.A_vld(mult_score_linear_A_vld),
			.A_dat(mult_score_linear_A_dat),
			.A_rdy(),
			.B_vld(mult_score_linear_B_vld),
			.B_dat(mult_score_linear_B_dat),
			.B_rdy(),
			.P_vld(mult_score_linear_P_vld),
			.P_dat(mult_score_linear_P_dat),
			.P_rdy(1'b1)
	);
  assign linear_rate_dat_out = mode? 0 : mult_score_linear_P_dat;
  assign linear_rate_vld_out = mode? 0 : mult_score_linear_P_vld;


  // linear_rate * avg_latency + BETA * (ddl - sys_clk)

  wire add_score_A_vld;
  wire add_score_B_vld;
  wire add_score_S_vld;

  wire [SCORE_BITWIDTH-1:0] add_score_A_dat;
  wire [SCORE_BITWIDTH-1:0] add_score_B_dat;
  wire [SCORE_BITWIDTH-1:0] add_score_S_dat;

  assign add_score_A_vld = mult_beta_P_vld;
  assign add_score_B_vld = mode? mult_score_linear_P_vld : 0;

  assign add_score_A_dat = mult_beta_P_dat;
  assign add_score_B_dat = mode? mult_score_linear_P_dat : 0;

	sif_addsub 
	# (
			.WIDTH(SCORE_BITWIDTH)
		) u_sif_addsub_score ( // SCORE_BITWIDTH bits input, SCORE_BITWIDTH bits outputs multipiler
			.clk(clk),
			.is_sub(1'b0),
			.A_vld(add_score_A_vld),
			.A_dat(add_score_A_dat),
			.A_rdy(),
			.B_vld(add_score_B_vld),
			.B_dat(add_score_B_dat),
			.B_rdy(),
			.S_vld(add_score_S_vld),
			.S_dat(add_score_S_dat),
			.S_rdy(1'b1)
	);

  assign score_vld = add_score_S_vld;
  assign score_dat = add_score_S_dat;

endmodule