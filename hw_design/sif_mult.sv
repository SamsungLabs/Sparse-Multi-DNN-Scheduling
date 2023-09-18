`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Design Name: 
// Module Name: sif_mult
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: WIDTH-bit multipiler
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module sif_mult
	#(
		parameter WIDTH = 32,
    parameter FIXED_POINT = 0
  )
  (
    input                       clk,
    input                       A_vld,
    input   [WIDTH-1:0]       A_dat,
    output                      A_rdy,
    input                       B_vld,
    input   [WIDTH-1:0]       B_dat,
    output                      B_rdy,
    output                      P_vld,
    output  [WIDTH-1:0]       P_dat,
    input                       P_rdy
  );
  wire  [2*WIDTH-1:0]       raw_P_dat;
  generate
    if((WIDTH == 32) && (FIXED_POINT != 1))  
    begin
      mult u_mult (
        .aclk(clk),                                  // input wire aclk
        .s_axis_a_tvalid(A_vld),            // input wire s_axis_a_tvalid
        //.s_axis_a_tready(A_rdy),            // output wire s_axis_a_tready
        .s_axis_a_tdata(A_dat),              // input wire [31 : 0] s_axis_a_tdata
        .s_axis_b_tvalid(B_vld),            // input wire s_axis_b_tvalid
        //.s_axis_b_tready(B_rdy),            // output wire s_axis_b_tready
        .s_axis_b_tdata(B_dat),              // input wire [31 : 0] s_axis_b_tdata
        .m_axis_result_tvalid(P_vld),  // output wire m_axis_result_tvalid
        //.m_axis_result_tready(P_rdy),  // input wire m_axis_result_tready
        .m_axis_result_tdata(P_dat)    // output wire [31 : 0] m_axis_result_tdata
      );     
    end
    else if((WIDTH == 16) && (FIXED_POINT != 1))  
    begin
      mult_fp16 u_mult_fp16 (
        .aclk(clk),                                  // input wire aclk
        .s_axis_a_tvalid(A_vld),            // input wire s_axis_a_tvalid
        //.s_axis_a_tready(A_rdy),            // output wire s_axis_a_tready
        .s_axis_a_tdata(A_dat),              // input wire [31 : 0] s_axis_a_tdata
        .s_axis_b_tvalid(B_vld),            // input wire s_axis_b_tvalid
        //.s_axis_b_tready(B_rdy),            // output wire s_axis_b_tready
        .s_axis_b_tdata(B_dat),              // input wire [31 : 0] s_axis_b_tdata
        .m_axis_result_tvalid(P_vld),  // output wire m_axis_result_tvalid
        //.m_axis_result_tready(P_rdy),  // input wire m_axis_result_tready
        .m_axis_result_tdata(P_dat)    // output wire [31 : 0] m_axis_result_tdata
      );     
    end
    else if((WIDTH == 32) && (FIXED_POINT == 1))  
    begin
      mult_fixed32 u_mult_fixed32 (
        .A(A_dat),      // input wire [31 : 0] A
        .B(B_dat),      // input wire [31 : 0] B
        .CLK(clk),  // input wire CLK
        .P(raw_P_dat)      // output wire [63 : 0] S
      );     
      assign P_dat = {raw_P_dat[2*WIDTH-1], raw_P_dat[48:18]}; // sign bit + 31 bits

      reg P_vld_r;
      always @(posedge clk) begin
        P_vld_r <= A_vld && B_vld;
      end
      assign P_vld = P_vld_r;

    end
    else if((WIDTH == 16) && (FIXED_POINT == 1))  
    begin
      mult_fixed16 u_mult_fixed16 (
        .A(A_dat),      // input wire [15 : 0] A
        .B(B_dat),      // input wire [15 : 0] B
        .CLK(clk),  // input wire CLK
        .P(raw_P_dat)      // output wire [31 : 0] S
      );
      assign P_dat = {raw_P_dat[2*WIDTH-1], raw_P_dat[22:8]}; // sign bit + 15 bits


      reg P_vld_r;
      always @(posedge clk) begin
        P_vld_r <= A_vld && B_vld;
      end
      assign P_vld = P_vld_r;
      
    end
  endgenerate 

endmodule