`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Design Name: 
// Module Name: fixed_to_float
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


module fixed_to_fp
	#(
		parameter WIDTH = 32
  )
  (
    input                       clk,
    input                       A_vld,
    input   [WIDTH-1:0]       A_dat,
    output                      A_rdy,
    output                      P_vld,
    output  [WIDTH-1:0]       P_dat,
    input                       P_rdy
  );

  generate
    if(WIDTH == 32)  
    begin
      fixed_to_32fp u_fixed_to_32fp (
        .aclk(clk),                                  // input wire aclk
        .s_axis_a_tvalid(A_vld),            // input wire s_axis_a_tvalid
        //.s_axis_a_tready(A_rdy),            // output wire s_axis_a_tready
        .s_axis_a_tdata(A_dat),              // input wire [31 : 0] s_axis_a_tdata
        .m_axis_result_tvalid(P_vld),  // output wire m_axis_result_tvalid
        //.m_axis_result_tready(P_rdy),  // input wire m_axis_result_tready
        .m_axis_result_tdata(P_dat)    // output wire [31 : 0] m_axis_result_tdata
      );     
    end
    else if(WIDTH == 16)  
    begin
      fixed_to_16fp u_fixed_to_16fp (
        .aclk(clk),                                  // input wire aclk
        .s_axis_a_tvalid(A_vld),            // input wire s_axis_a_tvalid
        //.s_axis_a_tready(A_rdy),            // output wire s_axis_a_tready
        .s_axis_a_tdata(A_dat),              // input wire [31 : 0] s_axis_a_tdata
        .m_axis_result_tvalid(P_vld),  // output wire m_axis_result_tvalid
        //.m_axis_result_tready(P_rdy),  // input wire m_axis_result_tready
        .m_axis_result_tdata(P_dat)    // output wire [31 : 0] m_axis_result_tdata
      );     
    end
  endgenerate 

endmodule