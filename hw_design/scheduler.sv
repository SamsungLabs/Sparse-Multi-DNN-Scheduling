//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Design Name: 
// Module Name: Dysta_Scheduler
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: Hardware implementation of Dysta Scheduler
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module Dysta_Scheduler
	#(
		parameter int X_dim = 3,
    parameter REQST_DEPTH = 10, // means 2^8 number of requests maximally
    parameter SCORE_BITWIDTH = 32, // 16 (FP16) or 32 (FP32)
    parameter COMPUTE_OPT = 1,
		parameter DATA_WIDTH = 16
    )
	(
		input clk,
		input reset,
		input [DATA_WIDTH-1:0] in_dat[X_dim-1:0], // pe data, in
    input in_vld,
    output [DATA_WIDTH-1:0] out_dat[X_dim-1:0], // pe data, out
		input compute_done_npu,
    input last_layer_done_npu,
		
    // Output to Dummy_NPU
    output [REQST_DEPTH-1:0] sel_task,
    output start_comp_npu,

		//	Ports for scheduler  //
		output scheduler_rdy,
		input [3-1:0] request_control,
		input [32-1:0] request_data
	);

  localparam REQST_RECEIVE = 1'b0;
  localparam REQST_UPDATE = 1'b1;

  localparam TASK_IDLE = 3'b000;
  localparam TASK_SELECT = 3'b001;
  localparam TASK_EXE = 3'b010;
  localparam TASK_CALC_SPARSE = 3'b011;
  localparam TASK_SCORING = 3'b100;
  localparam BETA = 3'b000;

  localparam MAX_LAYER = 50;
  localparam NUM_SPARSE_PATTERN = 4;

  // Assign outputs
  reg start_comp_npu_r;
  assign start_comp_npu = start_comp_npu_r;

  // LUT for different task informations
  reg     [SCORE_BITWIDTH-1:0] score_lut[(1 << REQST_DEPTH)-1:0];
  reg     [16-1:0] model_lut[(1 << REQST_DEPTH)-1:0];
  reg     [SCORE_BITWIDTH-1:0] ddl_lut[(1 << REQST_DEPTH)-1:0];
  reg     [SCORE_BITWIDTH-1:0] exe_clk_lut[(1 << REQST_DEPTH)-1:0];
  reg     [SCORE_BITWIDTH-1:0] linear_rate_lut[(1 << REQST_DEPTH)-1:0];
  reg     [32-1:0] sys_clk;

  always @(posedge clk or posedge reset)
  if(reset) begin
    sys_clk <= 0;
  end
  else begin
    sys_clk <= sys_clk + 1;
  end

  // Pass pe results from input and output
  reg [DATA_WIDTH-1:0] in_dat_r[X_dim-1:0];
  generate
  genvar i;
  for(i=0; i<X_dim; i++) 
    begin:gen_X
      always @(posedge clk or posedge reset)
      if(reset) begin
        in_dat_r[i] <= 0;
      end
      else begin
        in_dat_r[i] <= in_dat[i];
      end
      assign out_dat[i] = in_dat_r[i];
    end
  endgenerate

  // Count the num_zeros/sparsity in output. 
  // We assume the data is alredy in sparse storage formate, so only non-zeros will be outputed from PE.


  // LUT of average latency queue for each model and sparsity pattern

  // Lat LUT
  //  Address
  reg     [10-1:0] lat_lut_r_addr;
  reg     [10-1:0] lat_lut_w_addr;
  //  Enable  
  reg     lat_lut_w_enable;
  reg     lat_lut_r_enable;
  wire     lat_read_vld;
  //  Data
  wire     [SCORE_BITWIDTH-1:0] lat_read_dat;
  reg     [SCORE_BITWIDTH-1:0] lat_lut_w_dat;

  // Lat LUT
  //  Address
  reg     [10-1:0] sparsity_lut_r_addr;
  reg     [10-1:0] sparsity_lut_w_addr;
  //  Enable  
  reg     sparsity_lut_w_enable;
  reg     sparsity_lut_r_enable;
  wire    sparsity_read_vld;
  //  Data
  wire    [SCORE_BITWIDTH-1:0] sparsity_read_dat;
  reg     [SCORE_BITWIDTH-1:0] sparsity_lut_w_dat;

  // Lat LUT
  //  Address
  reg     [8-1:0] shape_lut_r_addr;
  reg     [8-1:0] shape_lut_w_addr;
  //  Enable  
  reg     shape_lut_w_enable;
  wire     shape_lut_r_enable;
  reg     shape_lut_r_enable_task;
  reg     shape_lut_r_enable_request;
  wire    shape_read_vld;
  //  Data
  wire    [16-1:0] shape_read_dat;
  reg     [16-1:0] shape_lut_w_dat;

  // Norm Isolation LUT
  //  Address
  reg     [10-1:0] norm_isol_lut_r_addr;
  reg     [10-1:0] norm_isol_lut_w_addr;
  //  Enable  
  reg     norm_isol_lut_w_enable;
  reg     norm_isol_lut_r_enable;
  wire    norm_isol_read_vld;
  //  Data
  wire    [16-1:0] norm_isol_read_dat;
  reg     [16-1:0] norm_isol_lut_w_dat;


  reg     [REQST_DEPTH-1:0] task_queue[(1 << REQST_DEPTH)-1:0];

  ram_simple_dual
  # (
    .w(SCORE_BITWIDTH),
    .d(1024)
  ) u_lat_lut
  (
    .clk(clk),  // common clock for read/write access
    .rst_n(!reset),

    .we(lat_lut_w_enable),   // active high write enable
    .write_addr(lat_lut_w_addr), 
    .din(lat_lut_w_dat),    // data in

    .re(lat_lut_r_enable),   // active high read enable
    .read_addr(lat_lut_r_addr),   // read address
    .dout_vld(lat_read_vld),
    .dout(lat_read_dat)     // data out
  ); // ram_simple_dual

  // LUT of sparsity queue for each model and sparsity pattern
  ram_simple_dual 
  # (
    .w(SCORE_BITWIDTH),
    .d(1024)
  ) u_sparsity_lut
  (
    .clk(clk),  // common clock for read/write access
    .rst_n(!reset),

    .we(sparsity_lut_w_enable),   // active high write enable
    .write_addr(sparsity_lut_w_addr), 
    .din(sparsity_lut_w_dat),    // data in

    .re(sparsity_lut_r_enable),   // active high read enable
    .read_addr(sparsity_lut_r_addr),   // read address
    .dout_vld(sparsity_read_vld),
    .dout(sparsity_read_dat)     // data out
  ); // ram_simple_dual

  assign shape_lut_r_enable = shape_lut_r_enable_task & shape_lut_r_enable_request;

  // LUT of output shape for each model
  ram_simple_dual 
  # (
    .w(16),
    .d(128)
  ) u_shape_lut
  (
    .clk(clk),  // common clock for read/write access
    .rst_n(!reset),
    .we(shape_lut_w_enable),   // active high write enable

    .write_addr(shape_lut_w_addr), 
    .din(shape_lut_w_dat),    // data in

    .re(shape_lut_r_enable),   // active high read enable
    .read_addr(shape_lut_r_addr),   // read address
    .dout_vld(shape_read_vld),
    .dout(shape_read_dat)     // data out
  ); // ram_simple_dual


  // LUT of norm isolation time for each model and sparsity pattern
  ram_simple_dual 
  # (
    .w(SCORE_BITWIDTH),
    .d(1024)
  ) u_norm_isol_lut
  (
    .clk(clk),  // common clock for read/write access
    .rst_n(!reset),

    .we(norm_isol_lut_w_enable),   // active high write enable
    .write_addr(norm_isol_lut_w_addr), 
    .din(norm_isol_lut_w_dat),    // data in

    .re(norm_isol_lut_r_enable),   // active high read enable
    .read_addr(norm_isol_lut_r_addr),   // read address

    .dout_vld(norm_isol_read_vld),
    .dout(norm_isol_read_dat)     // data out
  ); // ram_simple_dual


  // Control for task update
  reg     [3-1:0] task_state;
  reg     [SCORE_BITWIDTH-1:0] max_score;
  reg     sparse_update_done;
  reg     task_score_done;
  reg     [REQST_DEPTH-1:0] task_id_read;
  reg     [REQST_DEPTH-1:0] task_id_write;
  reg     [REQST_DEPTH-1:0] active_task;
  reg     [REQST_DEPTH-1:0] active_reqst_indx;
  reg     [16-1:0] num_nonzeros_r;
  wire    [16-1:0] num_nonzeros;

  reg     [REQST_DEPTH-1:0] num_reqst;
  reg     [REQST_DEPTH-1:0] num_read_task;
  reg     [REQST_DEPTH-1:0] num_write_task;
  reg     [16-1:0] task_info;
  reg     [10-1:0] read_addr;
  reg     [10-1:0] write_addr;
  reg     mode;

  always @(posedge clk or posedge reset)
  if(reset) begin
    task_state <= TASK_IDLE;
    start_comp_npu_r <= 0;
  end
  else begin
    if (task_state == TASK_IDLE) begin
      start_comp_npu_r <= 0;
      if ((num_reqst > 0) && (~request_control[0])) begin // Not incomming request
        task_state <= TASK_SELECT;
      end
    end 
    else if (task_state == TASK_SELECT) begin
      start_comp_npu_r <= 0;
      if (num_read_task == 0) begin
        task_state <= TASK_EXE;
      end
    end
    else if (task_state == TASK_EXE) begin
      start_comp_npu_r <= 1;
      if (compute_done_npu) begin
        task_state <= TASK_CALC_SPARSE;
      end
    end
    else if (task_state == TASK_CALC_SPARSE) begin
      start_comp_npu_r <= 0;
      if (sparse_update_done) begin
        task_state <= TASK_SCORING;
      end
    end
    else if (task_state == TASK_SCORING) begin
      start_comp_npu_r <= 0;
      if (task_score_done) begin
        task_state <= TASK_IDLE;
      end
    end
  end

  //    Compute Linear Rate
  wire      linear_rate_vld;
  wire      [SCORE_BITWIDTH-1:0] linear_rate_dat;
  assign    num_nonzeros = num_nonzeros_r;
  //    Compute Score
  reg     [SCORE_BITWIDTH-1:0] linear_rate_read;
  reg     [SCORE_BITWIDTH-1:0] ddl_read;
  reg     [SCORE_BITWIDTH-1:0] exe_clk_read;
  reg     [SCORE_BITWIDTH-1:0] norm_isolation_read;
  wire    score_vld;
  wire    [SCORE_BITWIDTH-1:0] score_dat;

  generate
    if(COMPUTE_OPT == 0)  
    begin
      Calc_Linear_Rate 
      #(
        .SCORE_BITWIDTH(SCORE_BITWIDTH)
      ) u_calc_linear_rate
      (
        .clk(clk),
        .reset(reset),

        .avg_sparsity_dat(sparsity_read_dat),
        .avg_sparsity_vld(sparsity_read_vld),

        .shape_dat(shape_read_dat),
        .shape_vld(shape_read_vld),

        .num_nonzeros(num_nonzeros),

        .linear_rate_dat_out(linear_rate_dat),
        .linear_rate_vld_out(linear_rate_vld)
      );

      Calc_Score 
      #(
        .SCORE_BITWIDTH(SCORE_BITWIDTH)
      ) u_calc_score
      (
        .clk(clk),
        .reset(reset),

        .avg_lat_dat(lat_read_dat),
        .avg_lat_vld(lat_read_vld),
        .linear_rate_dat_in(linear_rate_read),

        .exe_clk(exe_clk_read),
        .norm_isolation_vld(norm_isol_read_vld),
        .norm_isolation_dat(norm_isol_read_dat),

        .ddl(ddl_read),
        .sys_clk(sys_clk),

        .score_dat(score_dat),
        .score_vld(score_vld)
      );
    end
    else if(COMPUTE_OPT == 1)  
    begin
      Calc_Unit
      #(
        .SCORE_BITWIDTH(SCORE_BITWIDTH)
      ) u_calc_unit
      (
        .clk(clk),
        .reset(reset),
        .mode(mode),

        // Calc Linear Rate
        .avg_sparsity_dat(sparsity_read_dat),
        .avg_sparsity_vld(sparsity_read_vld),

        .shape_dat(shape_read_dat),
        .shape_vld(shape_read_vld),

        .num_nonzeros(num_nonzeros),
        .exe_clk(exe_clk_read),
        .norm_isolation_vld(norm_isol_read_vld),
        .norm_isolation_dat(norm_isol_read_dat),
        .linear_rate_dat_out(linear_rate_dat),
        .linear_rate_vld_out(linear_rate_vld),

        // Calc Linear Rate
        .avg_lat_dat(lat_read_dat),
        .avg_lat_vld(lat_read_vld),
        .linear_rate_dat_in(linear_rate_read),

        .ddl(ddl_read),
        .sys_clk(sys_clk),

        .score_dat(score_dat),
        .score_vld(score_vld)
      );
    end
  endgenerate 

  //    Action Control
  always @(posedge clk or posedge reset)
  if(reset) begin
    num_read_task <= 0;
    num_write_task <= 0;
    max_score <= 0;
    sparse_update_done <= 0;
    task_score_done <= 0;
    task_id_write <= 0;
    task_id_read <= 0;
    num_nonzeros_r <= 0;
    linear_rate_read <= 0;
    ddl_read <= 0;
    exe_clk_read <= 0;
    active_task <= 0;
    active_reqst_indx <= 0;
    mode <= 0;
  end
  else begin
    if (task_state == TASK_IDLE) begin
      num_read_task <= num_reqst;
      num_write_task <= 0;
      max_score <= 0;
      num_nonzeros_r <= 0;
      linear_rate_read <= 0;
      ddl_read <= 0;
      exe_clk_read <= 0;
      // read in advance to eliminate timing issue
      task_id_read <= task_queue[num_read_task-1];
      task_score_done <= 0;
      mode <= 0;
    end 
    else if (task_state == TASK_SELECT) begin
      if (num_read_task >= 0) begin
        if (max_score > score_lut[task_id_read]) begin
          active_task <= task_id_read;
          active_reqst_indx <= num_read_task;
          max_score <= score_lut[task_id_read];
          // Update task id
          task_id_read <= task_queue[num_read_task-1];
          // Update read time
          num_read_task <= num_read_task - 1;
        end
      end
    end
    else if (task_state == TASK_EXE) begin
      if (in_vld) num_nonzeros_r <= num_nonzeros_r + X_dim;
      // read in advance to eliminate timing issue
      num_read_task <= 1;
      task_info <=  model_lut[active_task];
      read_addr <= (task_info[4:0] * NUM_SPARSE_PATTERN + task_info[6:5]) * MAX_LAYER + model_lut[active_task][16-1:7];
    end
    else if (task_state == TASK_CALC_SPARSE) begin
        // Calc Address
        mode <= 1;
        if (num_read_task != 0) begin
          sparsity_lut_r_enable <= 1;
          sparsity_lut_r_addr <= read_addr;
          shape_lut_r_enable_task <= 1;
          shape_lut_r_addr <= read_addr;
          norm_isol_lut_r_enable <= 1;
          norm_isol_lut_r_addr <= read_addr;
          num_read_task <= num_read_task - 1;
        end 
        // Wait until linear rate is ready
        if (linear_rate_vld) begin
          sparsity_lut_r_enable <= 0;
          shape_lut_r_enable_task <= 0;
          norm_isol_lut_r_enable <= 0;
          linear_rate_lut[active_task] <= linear_rate_dat;
          sparse_update_done <= 1;
          num_read_task <= num_reqst;
          num_write_task <= num_reqst;
          // read in advance to eliminate timing issue
          task_id_read <= task_queue[num_read_task-1];
          task_info <=  model_lut[task_id_read];
          read_addr <= (task_info[4:0] * NUM_SPARSE_PATTERN + task_info[6:5]) * MAX_LAYER + model_lut[task_id_read][16-1:7];
        end
    end
    else if (task_state == TASK_SCORING) begin
      mode <= 0;
      if (num_write_task == 0) begin
        task_score_done <= 1;
      end
      else begin 
        if (num_write_task != 0) begin
          // Get lat table
          lat_lut_r_enable <= 1;
          lat_lut_r_addr <= read_addr;
          linear_rate_read <= linear_rate_lut[task_id_read];
          ddl_read <= ddl_lut[task_id_read];
          exe_clk_read <= exe_clk_lut[task_id_read];
          // Get task ID
          task_id_read <= task_queue[num_read_task-1];
          task_info <=  model_lut[task_id_read];
          read_addr <= (task_info[4:0] * NUM_SPARSE_PATTERN + task_info[6:5]) * MAX_LAYER + model_lut[task_id_read][16-1:7];
          // Update read time
          num_read_task <= num_read_task - 1;
        end

        // Update, Maybe some read/write delay is required to make the timing correct
        task_id_write <= task_queue[num_write_task-1];
        if (score_vld) begin
          // score_lut[task_id_write] <= score_dat;
          num_write_task <= num_write_task - 1;
        end
      end
    end
  end
  assign sel_task = active_task;

  // Control for request update

  reg     reqst_state;
  reg     scheduler_rdy_r;
  reg     [REQST_DEPTH-1:0] reqst_id;

  //    State Control
  always @(posedge clk or posedge reset)
  if(reset) begin
    reqst_state <= REQST_RECEIVE;
  end
  else begin
    if (task_state == TASK_SCORING) begin
      reqst_state <= REQST_UPDATE;
    end 
    else begin
      reqst_state <= REQST_RECEIVE;
    end
  end

  //    Action Control

  always @(posedge clk or posedge reset)
  if(reset) begin
    scheduler_rdy_r <= 0;
    num_reqst <= 0;
    reqst_id <= 0;
    write_addr <= 0;
    lat_lut_w_enable <= 1'b0;
    sparsity_lut_w_enable  <= 1'b0;
    shape_lut_r_enable_request  <= 1'b0;
    lat_lut_w_addr <= 0;
    sparsity_lut_w_addr <= 0;
    shape_lut_w_addr <= 0;
    lat_lut_w_dat <= 0;
    sparsity_lut_w_dat <= 0;
    norm_isol_lut_w_dat <= 0;
    shape_lut_w_dat <= 0;
  end
  else begin
    lat_lut_w_enable <= 1'b0;
    sparsity_lut_w_enable  <= 1'b0;
    shape_lut_r_enable_request  <= 1'b0;
    lat_lut_w_dat <= 0;
    sparsity_lut_w_dat <= 0;
    norm_isol_lut_w_dat <= 0;
    shape_lut_w_dat <= 0;
    // Update exe time whichever the state is
    if (compute_done_npu) begin
      exe_clk_lut[active_task] <= sys_clk;
    end
    // Finite State Machine Control
    if (reqst_state == REQST_RECEIVE) begin
      scheduler_rdy_r <= 1;
      if (request_control == 4'b1000) begin // receive model/sparsity information
        reqst_id <= request_data[(REQST_DEPTH+7)-1:7];
        model_lut[request_data[(REQST_DEPTH+7)-1:7]][0:6] <= request_data[6:0]; // 0:4 model ID, 5:6 sparsity pattern ID
      end
      else if (request_control == 4'b1001) begin // receive deadline information 
        ddl_lut[reqst_id] <= request_data; // ddl information
        exe_clk_lut[reqst_id] <= sys_clk; // ddl information
      end 
      else if (request_control == 4'b1010) begin // receive initial score information 
        score_lut[reqst_id] <= request_data;
        task_queue[num_reqst] <= reqst_id;
        num_reqst <= num_reqst + 1;
      end 
      // Update Lat/Sparsity/Shape LUT
      else if (request_control == 4'b0001) begin // Get initial address
        reqst_id <= request_data[(REQST_DEPTH+7)-1:7];
        write_addr <= (request_data[4:0] * NUM_SPARSE_PATTERN + request_data[6:5]) * MAX_LAYER;
      end 
      else if (request_control == 4'b0010) begin // update latency LUT 
        lat_lut_w_enable <= 1'b1;
        sparsity_lut_w_enable  <= 1'b0;
        norm_isol_lut_w_enable <= 1'b0;
        shape_lut_r_enable_request  <= 1'b0;
        lat_lut_w_dat <= request_data[REQST_DEPTH-1:0];
        lat_lut_w_addr <= write_addr;
      end 
      else if (request_control == 4'b0011) begin // update sparsity LUT  
        lat_lut_w_enable <= 1'b0;
        sparsity_lut_w_enable  <= 1'b1;
        norm_isol_lut_w_enable <= 1'b0;
        shape_lut_r_enable_request  <= 1'b0;
        sparsity_lut_w_dat <= request_data[REQST_DEPTH-1:0];
        sparsity_lut_w_addr <= write_addr;
      end 
      else if (request_control == 4'b0100) begin // update norm isolation LUT 
        lat_lut_w_enable <= 1'b0;
        sparsity_lut_w_enable  <= 1'b0;
        shape_lut_r_enable_request  <= 1'b0;
        norm_isol_lut_w_enable <= 1'b1;
        norm_isol_lut_w_dat <= request_data[REQST_DEPTH-1:0];
        norm_isol_lut_w_addr <= write_addr;
      end 
      else if (request_control == 4'b0101) begin // update shape LUT 
        lat_lut_w_enable <= 1'b0;
        sparsity_lut_w_enable  <= 1'b0;
        norm_isol_lut_w_enable <= 1'b0;
        shape_lut_r_enable_request  <= 1'b1;
        shape_lut_w_dat <= request_data[REQST_DEPTH-1:0];
        shape_lut_w_addr <= write_addr;
        write_addr <= write_addr + 1; // increase layer
      end 
    end 
    else if (reqst_state == REQST_UPDATE) begin
      scheduler_rdy_r <= 0;
      if (last_layer_done_npu && task_score_done) begin
        task_queue[active_reqst_indx] <= task_queue[num_reqst];
        num_reqst <= num_reqst - 1;
      end
      if (score_vld) begin
        score_lut[task_id_write] <= score_dat;
      end
    end
  end
  assign scheduler_rdy_r = scheduler_rdy;

endmodule