`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Design Name: 
// Module Name: sif_div
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: Input 16 bits interger, output WIDTH-bits fp
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module sif_div
	#(
		parameter WIDTH = 32
  )
  (
  input                       clk,
  input                       A_vld,
  input   [16-1:0]       A_dat,
  output                      A_rdy,
  input                       B_vld,
  input   [16-1:0]       B_dat,
  output                      B_rdy,
  output                      P_vld,
  output  [WIDTH-1:0]       P_dat,
  input                       P_rdy
    );

  // get output
  generate
    if(WIDTH == 32)  
    begin
      div u_div (
        .aclk(clk),                                  // input wire aclk
        .s_axis_divisor_tvalid(A_vld),            // input wire s_axis_a_tvalid
        // .s_axis_divisor_tready(A_rdy),            // output wire s_axis_a_tready
        .s_axis_divisor_tdata(A_dat),              // input wire [15 : 0] s_axis_a_tdata
        .s_axis_dividend_tvalid(B_vld),            // input wire s_axis_b_tvalid
        // .s_axis_dividend_tready(B_rdy),            // output wire s_axis_b_tready
        .s_axis_dividend_tdata(B_dat),              // input wire [15 : 0] s_axis_b_tdata
        .m_axis_dout_tvalid(P_vld),  // output wire m_axis_result_tvalid
        // .m_axis_dout_tready(P_rdy),  // input wire m_axis_result_tready
        .m_axis_dout_tdata(P_dat)    // output wire [31 : 0] m_axis_result_tdata
      );  
    end
    else if(WIDTH == 16)  
    begin
      div u_div_fp16 (
        .aclk(clk),                                  // input wire aclk
        .s_axis_divisor_tvalid(A_vld),            // input wire s_axis_a_tvalid
        // .s_axis_divisor_tready(A_rdy),            // output wire s_axis_a_tready
        .s_axis_divisor_tdata(A_dat),              // input wire [15 : 0] s_axis_a_tdata
        .s_axis_dividend_tvalid(B_vld),            // input wire s_axis_b_tvalid
        // .s_axis_dividend_tready(B_rdy),            // output wire s_axis_b_tready
        .s_axis_dividend_tdata(B_dat),              // input wire [15 : 0] s_axis_b_tdata
        .m_axis_dout_tvalid(P_vld),  // output wire m_axis_result_tvalid
        // .m_axis_dout_tready(P_rdy),  // input wire m_axis_result_tready
        .m_axis_dout_tdata(P_dat)    // output wire [31 : 0] m_axis_result_tdata
      ); 
    end
  endgenerate 




endmodule