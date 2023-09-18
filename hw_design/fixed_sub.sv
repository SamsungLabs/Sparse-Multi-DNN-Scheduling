`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Design Name: 
// Module Name: fixed_sub
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: WIDTH-bit sub, fixed point
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module fixed_sub
	#(
		parameter WIDTH = 32
  )
  (
    input                       clk,
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
  reg S_vld_r;

	always @(posedge clk) begin
    S_vld_r <= A_vld && B_vld;
  end
  assign S_vld = S_vld_r;

  // get output
  generate
    if(WIDTH == 32)  
    begin
      sub_fixed32 u_sub_fixed32 (
        .A(A_dat),      // input wire [31 : 0] A
        .B(B_dat),      // input wire [31 : 0] B
        .CLK(clk),  // input wire CLK
        .S(S_dat)      // output wire [31 : 0] S
      );    
    end
    else if(WIDTH == 16)  
    begin
      sub_fixed16 u_sub_fixed16 (
        .A(A_dat),      // input wire [15 : 0] A
        .B(B_dat),      // input wire [15 : 0] B
        .CLK(clk),  // input wire CLK
        .S(S_dat)      // output wire [15 : 0] S
      );    
    end
  endgenerate 

endmodule