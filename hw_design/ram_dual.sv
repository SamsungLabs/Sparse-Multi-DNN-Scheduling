//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Design Name: 
// Module Name: ram_dual
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module ram_simple_dual
# (
  parameter is_lut = 1,
  parameter w = 128,
  parameter d = 128
)
(
  input                   clk,  // common clock for read/write access
  input                   rst_n,
  input                   we,   // active high write enable
  // input   [$clog2(d)-1:0] write_addr, 
  input   [32-1:0] write_addr, 
  input   [w-1:0]         din,    // data in

  input                   re,   // active high read enable
  input   [32-1:0] read_addr,   // read address
  output                   dout_vld,
  output  [w-1:0]         dout     // data out
); // ram_simple_dual

  reg                 dout_vld_r;

  always @(posedge clk or negedge rst_n)
  if(!rst_n) begin
      dout_vld_r <= 1'b0;
  end
  else begin
      dout_vld_r <= re;
  end

  assign dout_vld = dout_vld_r;

  // get output
  generate
    if(w == 32 && d == 1024 && is_lut == 0)  
    begin
        ram_naive_32_1024_1r1w u_ram_naive_32_1024_1r1w (
        .clka(clk),    // input wire clka
        .ena(we),      // input wire ena // use as write port
        .wea(we),      // input wire [0 : 0] wea
        .addra(write_addr[$clog2(d)-1:0]),  // input wire [6 : 0] addra
        .dina(din),    // input wire [127 : 0] dina
        .clkb(clk),    // input wire clkb
        .enb(re),      // input wire enb
        .addrb(read_addr[$clog2(d)-1:0]),  // input wire [6 : 0] addrb
        .doutb(dout)  // output wire [127 : 0] doutb
        );
    end
    else if(w == 16 && d == 128 && is_lut == 0)  
    begin
        ram_naive_16_128_1r1w u_ram_naive_16_128_1r1w (
        .clka(clk),    // input wire clka
        .ena(we),      // input wire ena // use as write port
        .wea(we),      // input wire [0 : 0] wea
        .addra(write_addr[$clog2(d)-1:0]),  // input wire [6 : 0] addra
        .dina(din),    // input wire [127 : 0] dina
        .clkb(clk),    // input wire clkb
        .enb(re),      // input wire enb
        .addrb(read_addr[$clog2(d)-1:0]),  // input wire [6 : 0] addrb
        .doutb(dout)  // output wire [127 : 0] doutb
        );
    end
    else if(w == 16 && d == 1024 && is_lut == 0)  
    begin
        ram_naive_16_1024_1r1w u_ram_naive_16_1024_1r1w (
        .clka(clk),    // input wire clka
        .ena(we),      // input wire ena // use as write port
        .wea(we),      // input wire [0 : 0] wea
        .addra(write_addr[$clog2(d)-1:0]),  // input wire [6 : 0] addra
        .dina(din),    // input wire [127 : 0] dina
        .clkb(clk),    // input wire clkb
        .enb(re),      // input wire enb
        .addrb(read_addr[$clog2(d)-1:0]),  // input wire [6 : 0] addrb
        .doutb(dout)  // output wire [127 : 0] doutb
        );
    end
    else if(is_lut == 1)  
    begin
        lut_dual 
        #(
          .DATA_BITWIDTH(w),
          .ADDR_BITWIDTH($clog2(d))
        ) u_lut_dual (
        .clk(clk),    // input wire clka
        .read_req(re),
        .write_en(we),
        .w_addr(write_addr[$clog2(d)-1:0]),
        .r_addr(read_addr[$clog2(d)-1:0]),
        .w_data(din),
        .r_data(dout)
        );
    end
  endgenerate 

endmodule // ram_simple_dual