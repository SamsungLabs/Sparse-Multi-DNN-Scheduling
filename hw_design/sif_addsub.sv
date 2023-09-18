`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Design Name: 
// Module Name: sif_addsub
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: WIDTH-bit adder/subber, controlled by is_sub
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module sif_addsub
	#(
		parameter WIDTH = 32
  )
  (
    input                       clk,
    input                       is_sub,
    input                       A_vld,
    input   [WIDTH-1:0]       A_dat,
    output                      A_rdy,
    input                       B_vld,
    input   [WIDTH-1:0]       B_dat,
    output                      B_rdy,
    output                      S_vld,
    output  [WIDTH-1:0]       S_dat,
    input                       S_rdy
  );

  // get output
  generate
    if(WIDTH == 32)  
    begin
      addsub u_addsub (
        .aclk(clk),                                  // input wire aclk
        .s_axis_a_tvalid(A_vld),            // input wire s_axis_a_tvalid
        //.s_axis_a_tready(A_rdy),            // output wire s_axis_a_tready
        .s_axis_a_tdata(A_dat),              // input wire [31 : 0] s_axis_a_tdata
        .s_axis_b_tvalid(B_vld),            // input wire s_axis_b_tvalid
        //.s_axis_b_tready(B_rdy),            // output wire s_axis_b_tready
        .s_axis_b_tdata(B_dat),              // input wire [31 : 0] s_axis_b_tdata
        .s_axis_operation_tvalid(1'b1),  // input wire s_axis_operation_tvalid
        //.s_axis_operation_tready(),  // output wire s_axis_operation_tready
        .s_axis_operation_tdata({{7{1'b0}}, is_sub}),    // input wire [7 : 0] s_axis_operation_tdata
        .m_axis_result_tvalid(S_vld),  // output wire m_axis_result_tvalid
        //.m_axis_result_tready(S_rdy),  // input wire m_axis_result_tready
        .m_axis_result_tdata(S_dat)    // output wire [31 : 0] m_axis_result_tdata
      );    
    end
    else if(WIDTH == 16)  
    begin
      addsub_fp16 u_addsub_16fp (
        .aclk(clk),                                  // input wire aclk
        .s_axis_a_tvalid(A_vld),            // input wire s_axis_a_tvalid
        //.s_axis_a_tready(A_rdy),            // output wire s_axis_a_tready
        .s_axis_a_tdata(A_dat),              // input wire [31 : 0] s_axis_a_tdata
        .s_axis_b_tvalid(B_vld),            // input wire s_axis_b_tvalid
        //.s_axis_b_tready(B_rdy),            // output wire s_axis_b_tready
        .s_axis_b_tdata(B_dat),              // input wire [31 : 0] s_axis_b_tdata
        .s_axis_operation_tvalid(1'b1),  // input wire s_axis_operation_tvalid
        //.s_axis_operation_tready(),  // output wire s_axis_operation_tready
        .s_axis_operation_tdata({{7{1'b0}}, is_sub}),    // input wire [7 : 0] s_axis_operation_tdata
        .m_axis_result_tvalid(S_vld),  // output wire m_axis_result_tvalid
        //.m_axis_result_tready(S_rdy),  // input wire m_axis_result_tready
        .m_axis_result_tdata(S_dat)    // output wire [31 : 0] m_axis_result_tdata
      );    
    end
  endgenerate 

endmodule