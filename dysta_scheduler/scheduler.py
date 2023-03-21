from utils import Task, take
import logging
import numpy as np
from dysta_lat_pred import avg_pred_linear_rate, last_one_pred_linear_rate, last_N_pred_linear_rate

class Scheduler:
  def __init__(self, reqst_table):
    self.sys_time = 0.0
    self.reqst_table = reqst_table # [(req_time1, target_lat1, model1), (req_time2, target_lat2, model2), ....]
    self.cur_reqst_indx = 0
    self.running_task = {} # {{task_id1: task1}, {task_id2: task2} ...}
    self.finished_reqst = {}
    self.num_reqst= len(self.reqst_table)
    self.lat_lut = {}

  def is_finished(self):
    """
    Checks that all input requests have been scheduled. 
    """
    return len(self.finished_reqst) == self.num_reqst

  def set_lat_lut(self, lat_lut):
    self.lat_lut = lat_lut

  def calc_violation_rate(self):
    assert self.is_finished() # Check if all finished
    num_violate_tasks = 0
    violate_task_dict = {} # [(reqst, target, model_str), ....]
    for task_id, task_info in self.finished_reqst.items():
      if task_info.finish_time > task_info.target_time:
        num_violate_tasks += 1
        violate_task_dict[task_info.reqst_time] = {"reqst_time:": task_info.reqst_time, 
            "target_time:": task_info.target_time, "finsh_time:": task_info.finish_time, "task_id": task_id}
    # Sort the violate task dict according to request time
    violate_task_dict = {k: v for k, v in sorted(violate_task_dict.items(), key=lambda item: item[0])}
    violation_rate = num_violate_tasks/self.num_reqst
    assert (violation_rate >= 0) and (violation_rate <= 1)
    return violation_rate, violate_task_dict

  def calc_system_thrpt(self):
    """
    Returns the system throughput (STP) in inferences per second (inf/s).
    """
    assert self.is_finished()
    last_finish_time = 0
    for task_id,task_info in self.finished_reqst.items():
      if last_finish_time <= task_info.finish_time:
        last_finish_time = task_info.finish_time
    first_reqst_time, _, _, _, _, _, _ = self.reqst_table[0]
    exec_time_total = last_finish_time - first_reqst_time
    system_thrpt = self.num_reqst / exec_time_total
    
    assert system_thrpt >= 0
    return system_thrpt

  def calc_ANTT(self):
    """
    Returns the average normalised turnaround time (ATNN) 
    across all inference tasks. This metric captures the average slowdown
    with respect to the isolated execution of its task on the same hardware.
    """
    assert self.is_finished()
    task_ntts = []
    for task_id,task_info in self.finished_reqst.items():
      task_exec_time = task_info.finish_time - task_info.reqst_time
      task_ntts.append(task_exec_time / task_info.real_isolated_time)

    antt = np.mean(task_ntts)
    return antt

  def __push_reqst(self, reqst_indx):
    # Read the new input request
    reqst_time, target_lat, reqst_model, priority, avg_lat, sample_id, avg_sparsity = self.reqst_table[reqst_indx]
    
    # Construct a new task
    new_task = Task(reqst_time, target_lat, reqst_model, priority, avg_lat, avg_sparsity)


    new_task.construct_task(self.lat_lut[reqst_model]['lat_lut'][sample_id], self.lat_lut[reqst_model]['sparsity_lut'][sample_id])
    task_id = str(reqst_time) + "_" + reqst_model # log using arrival time and model name
    
    # Add task into running tasks
    assert task_id not in self.running_task
    # print ("push reqst:", task_id)
    self.running_task[task_id] = new_task

  def update_reqst(self):
    if (self.cur_reqst_indx < self.num_reqst): 
      # request info: req_time, target_lat, model, priority
      reqst_time, target_lat, reqst_model, priority, avg_lat, sample_id, avg_sparsity = self.reqst_table[self.cur_reqst_indx]
      
      # No running task, push the nearest request
      if (len(self.running_task) == 0): 
        self.__push_reqst(self.cur_reqst_indx)
        self.cur_reqst_indx += 1
        self.sys_time = reqst_time

      # Push all the requests behind the sys_time
      else:
        while (reqst_time < self.sys_time):
          self.__push_reqst(self.cur_reqst_indx)
          
          # Update request info
          self.cur_reqst_indx += 1
          if (self.cur_reqst_indx >= self.num_reqst): break
          reqst_time, target_lat, reqst_model, priority, avg_lat, sample_id, avg_sparsity = self.reqst_table[self.cur_reqst_indx]
    else:
      pass
  
  # Virtual class - to be implemented by each scheduler
  def update_schedule(self):
    raise NotImplementedError()

  def exe(self, is_hw_monitor=False): # Per-layer granularity
    task_id = self.update_schedule() # id of the scheduled task
    lat_consumed = self.running_task[task_id].exe(is_hw_monitor) # latency of the executed layer
    self.sys_time += lat_consumed
    self.running_task[task_id].last_exe_time = self.sys_time
    # Check any finished task
    if (self.running_task[task_id].is_finished(self.sys_time)):
      self.finished_reqst[task_id] = self.running_task[task_id]
      del self.running_task[task_id]
      # print ("finish reqst:", task_id)

class FCFS_Scheduler(Scheduler):
  """
  This scheduler imposes a first-come first-served (FCFS) approach.
  """
  def __init__(self,reqst_table):
    super().__init__(reqst_table)
    print ("Constructing FCFS Scheduler.")

  def reset(self, reqst_table):
    self.__init__(reqst_table)

  def update_schedule(self):
    next_task_id = None

    # Find the earliest task in the queue
    earliest_reqst_time = -1
    for task_id,task in self.running_task.items():
      if ((earliest_reqst_time < 0) or (earliest_reqst_time > task.reqst_time)): 
        next_task_id = task_id
        earliest_reqst_time = task.reqst_time

    logging.debug("next task:%s, sys time:%f" % (next_task_id, self.sys_time))
    return next_task_id
    # raise NotImplementedError()


class PREMA_Scheduler(Scheduler):
  """
  This scheduler implements PREMA (HPCA 2020).
  """
  def __init__(self,reqst_table, is_sparse=False):
    super().__init__(reqst_table)
    if is_sparse:
      print ("Constructing Sparse PREMA Scheduler.")
    else:
      print ("Constructing PREMA Scheduler.")
    self.ready_queue = {}
    self.is_sparse = is_sparse

  def reset(self, reqst_table):
    self.__init__(reqst_table)

  def update_schedule(self):
    next_task_id = None

    # Update token score
    max_prema_token = 0
    max_task_id = None
    for task_id,task in self.running_task.items():
      
      # Initialise tokens with priorities
      if (task.prema_token < 0): 
        task.prema_token = task.priority # Line 3 in PREMA paper, Algorithm 2
        max_task_id = task_id
      else: 
        # Line 7 in PREMA paper, Algorithm 2
        idle_time =  self.sys_time - task.last_exe_time
        if self.is_sparse:
          slowdown_rate = idle_time / task.real_isolated_time # TODO - latest idle time or overall waiting time?
        else:
          slowdown_rate = idle_time / task.prema_est_isolated_time
        task.prema_token += task.priority * slowdown_rate
        if max_prema_token < task.prema_token:
          max_prema_token = task.prema_token
          max_task_id = task_id
    # Get threshold based on the max threshold, 
    # according to the paragragh under TABLE II in the PREMA paper
    threshold = -1
    if (max_prema_token >= 9): 
      threshold = 9
    elif (max_prema_token >= 3): 
      threshold = 3
    else: 
      threshold = 1
    
    # Get candidate tasks, line 10 in PREMA paper, Algorithm 2
    candidate_tasks = []
    for task_id,task in self.running_task.items():
      ###################!!!!!!!We change it to "Token >= Threshold", or it does not make sense at the beginning!!!!!!!!!###########
      if (task.prema_token >= threshold): candidate_tasks.append(task_id)
    
    # Get next token using shortest estimated time approach
    logging.debug("threshold:%d" % (threshold))
    shortest_time = -1
    for task_id in candidate_tasks:
      if self.is_sparse:
        estimated_time = sum(self.running_task[task_id].real_lat_queue) # Use real sparsity to estimate lat
      else:
        estimated_time = sum(self.running_task[task_id].est_lat_queue) # Use estimate avg (PREMA) lat  
      if len(candidate_tasks) > 1:
        logging.debug("task in candidate:%s with estimated time:%f" % (task_id, estimated_time))
      if ((next_task_id is None) or (shortest_time > estimated_time)): 
        next_task_id = task_id
        shortest_time = estimated_time
    logging.debug("next task:%s, sys time:%f" % (next_task_id, self.sys_time))
    return next_task_id
    # logging.debug("next task:%s, sys time:%f" % (next_task_id, self.sys_time))

class Dysta_Scheduler(Scheduler):
  """
  This scheduler implements our HW-SW co-design approach with sparse latency predictor used while scheduling.
  Used to simulate the real hardware execution of our approach.
  """
  def __init__(self,reqst_table, penalty_eff=1.0, num_candidate=5, beta=0.01):
    super().__init__(reqst_table)
    print ("Constructing Dysta HW/SW Scheduler.")
    self.penalty_eff = penalty_eff
    self.num_candidate = num_candidate
    self.beta = beta # Parameter used to control weighting of each metric

  def reset(self, reqst_table):
    self.__init__(reqst_table)

  def update_schedule(self):
    next_task_id = None
    for task_id,task in self.running_task.items():
      # Update gamma
      num_exe_layer = len(task.dysta_measured_sparsities)
      if (num_exe_layer > 0):
        # Last one predictor because: (Check dysta_lat_pred.py for details.) 
        #     1. it has the lower RMSE
        #     2. Lower resource consumption (Momeory and computation).
        # task.gamma = avg_pred_linear_rate(task.dysta_measured_sparsities, task.dysta_avg_sparsities)
        # task.gamma = last_N_pred_linear_rate(task.dysta_measured_sparsities, task.dysta_avg_sparsities)
        task.gamma = last_one_pred_linear_rate(task.dysta_measured_sparsities, task.dysta_avg_sparsities)

      # Calculate urgency
      slack_time =  task.target_time - self.sys_time
      torun_lat = sum(task.real_lat_queue)
      task.dysta_urgency = slack_time - torun_lat
      if (task.dysta_urgency < 0): task.dysta_urgency = 0

      # Calculate preemption penalty, two purpose: 
      #     1. Avoid task switch as it cause extra resource consumption; 
      #     2. Encourage to resume the last execute task as it has higher possibility to yeild higher ANTT
      idle_time =  self.sys_time - task.last_exe_time
      penalty_preemption = idle_time / task.real_isolated_time # TODO - latest idle time or overall waiting time?
      # Normalize preemption by the number of processes
      penalty_preemption /= len(self.running_task)
      task.dysta_score = task.dysta_urgency + penalty_preemption # - penalty_vio

    # Get next token according to shorted estimated time
    shortest_time = -1
    for task_id, task in self.running_task.items():
      estimated_time = task.dysta_gamma * sum(task.est_lat_queue) # Use real sparsity to estimate lat
      estimated_time = estimated_time + self.beta * task.dysta_score # / task.priority
      logging.debug("choose candidate:%s, score:%f"%(task_id, estimated_time))
      if ((next_task_id is None) or (shortest_time > estimated_time)): 
        next_task_id = task_id
        shortest_time = estimated_time 
        
    logging.debug("next task of dysta_oracle:%s, sys time:%f" % (next_task_id, self.sys_time))
    return next_task_id



class Dysta_Oracle_Scheduler(Scheduler):
  """
  This scheduler implements our scheduling approach based on oracle latency information.
  Used to investigate what is the optimal attainable performance.
  """
  def __init__(self,reqst_table, penalty_eff=1.0, num_candidate=5, beta=0.01):
    super().__init__(reqst_table)
    print ("Constructing Dysta Scheduler that uses oracle latency information for scheduling.")
    self.penalty_eff = penalty_eff
    self.num_candidate = num_candidate
    self.beta = beta # Parameter used to control weighting of each metric

  def reset(self, reqst_table):
    self.__init__(reqst_table)

  def cal_violate_rate(self, sys_time, exe_task_id=None):
    """
    Calculate violation rate given the current sys_time.
    """
    num_running_task = len(self.running_task)
    num_violate_tasks = 0
    for task_id,task in self.running_task.items():
      est_lat = sum(self.running_task[task_id].real_lat_queue)
      est_finish_time = sys_time + est_lat
      if (est_finish_time > self.running_task[task_id].target_time):
        if (exe_task_id==None) or (task_id != exe_task_id):
          num_violate_tasks += 1 
    vio_rate = num_violate_tasks / num_running_task
    return vio_rate

  def update_schedule(self):
    next_task_id = None
    cur_vio_rate = self.cal_violate_rate(self.sys_time)
    task_score_list = {}
    for task_id,task in self.running_task.items():
      # Calculate violation penalty
      # Comment out because these metrics do not help. But keep this, in case we need it in the furture.
      '''
      lookahead_sys_time = self.sys_time + self.running_task[task_id].real_lat_queue[0]  # Add the exe time of the next layer on top ofsys_time 
      lookahead_vio_rate = self.cal_violate_rate(lookahead_sys_time, task_id)
      dif_vio_rate = lookahead_vio_rate - cur_vio_rate
      penalty_vio = self.penalty_eff * dif_vio_rate
      '''

      # Calculate urgency
      slack_time =  task.target_time - self.sys_time
      torun_lat = sum(task.real_lat_queue)
      task.dysta_urgency = slack_time - torun_lat
      if (task.dysta_urgency < 0): task.dysta_urgency = 0

      # Calculate preemption penalty, two purposes: 
      #     1. Avoid task switch as it can cause extra resource consumption.
      #     2. Encourage resuming the already running task as it has higher probability to yield higher ANTT.
      idle_time =  self.sys_time - task.last_exe_time
      penalty_preemption = idle_time / task.real_isolated_time # TODO - latest idle time or overall waiting time?
      # Normalize preemption by the number of processes
      penalty_preemption /= len(self.running_task)
      task.dysta_score = task.dysta_urgency + penalty_preemption # - penalty_vio

      # Get average score
      # Comment out because choosing from candidate does not help. But keep this in case we need it in the furture.
      '''
      task_score = task.dysta_urgency - penalty_vio + slowdown_rate
      task_score_list[task_id] = task_score
      logging.debug("task_id:%s, task_score:%f, task_ddl:%f, lookahead_sys_time:%f, remain_time:%f, torun_time:%f"%(task_id, task_score, task.target_time, lookahead_sys_time, slack_time, torun_lat))
      logging.debug("task.dysta_urgency:%f, penalty_vio_rate:%f, slowdown_rate:%f" % (task.dysta_urgency, penalty_vio, slowdown_rate))
      '''
    # Sort the task list by score 
    # Comment out because choosing from candidate does not help. But keep this, in case we need it in the furture.
    '''
    sorted_task_score_list = {k: v for k, v in sorted(task_score_list.items(), key=lambda item: item[1], reverse=True)}
    candidate_tasks = {}
    for k, v in sorted_task_score_list.items():
      if len(candidate_tasks) >= self.num_candidate: break
      else: 
        candidate_tasks[k] = v
    '''

    # Get next token according to shorted estimated time
    shortest_time = -1
    for task_id, task in self.running_task.items():
      estimated_time = sum(self.running_task[task_id].real_lat_queue) # Use real sparsity to estimate lat
      estimated_time = estimated_time + self.beta * task.dysta_score # / task.priority
      logging.debug("choose candidate:%s, score:%f"%(task_id, estimated_time))
      if ((next_task_id is None) or (shortest_time > estimated_time)): 
        next_task_id = task_id
        shortest_time = estimated_time 
        
    logging.debug("next task of dysta_oracle:%s, sys time:%f" % (next_task_id, self.sys_time))
    return next_task_id

class SDRM3_Scheduler(Scheduler):
  """
  This scheduler implements SDRM3 (arXiv 2022).
  """
  def __init__(self,reqst_table, alpha=1.0):
    super().__init__(reqst_table)
    self.alpha = alpha
    print ("Constructing SDRM3 Scheduler.")

  def reset(self, reqst_table):
    self.__init__(reqst_table)

  def update_schedule(self):
    next_task_id = None
    highest_urgency = -1
    max_map_score = -1
    for task_id,task in self.running_task.items():
      
      # Calculate urgency
      slack_time =  task.target_time - self.sys_time
      torun_lat = sum(task.est_lat_queue)
      task.sdrm_urgency = torun_lat/slack_time
      task.sdrm_urgency = 1 if task.sdrm_urgency > 1 else task.sdrm_urgency

      # Calculate fairness
      idle_time =  self.sys_time - task.last_exe_time
      task.fairness = idle_time / task.est_lat_queue[0]

      # Calculate MapScore
      task.map_score = task.sdrm_urgency + self.alpha * task.fairness

      # if ((highest_urgency < 0) or (highest_urgency <= task.sdrm_urgency)): 
      if ((max_map_score < 0) or (max_map_score <= task.map_score)):
        # if ((highest_urgency == task.sdrm_urgency) and (self.running_task[next_task_id].reqst_time < task.reqst_time)):
        if ((max_map_score == task.map_score) and (self.running_task[next_task_id].reqst_time < task.reqst_time)):
          # If with the same, use FCFS
          continue
        else:
          next_task_id = task_id
          highest_urgency = task.sdrm_urgency
          max_map_score = task.map_score
    logging.debug("next task:%s, sys time:%f" % (next_task_id, self.sys_time))
    return next_task_id

class SJF_Scheduler(Scheduler):
  """
  This scheduler implements Shortest Estimated Job First (SJF).
  """
  def __init__(self,reqst_table, is_sparse = False):
    super().__init__(reqst_table)
    self.is_sparse = is_sparse
    print ("Constructing SJF Scheduler.")

  def reset(self, reqst_table):
    self.__init__(reqst_table)

  def update_schedule(self):
    next_task_id = None
    shortest_time = -1
    for task_id, task in self.running_task.items():
      if self.is_sparse:
        estimated_time = sum(self.running_task[task_id].real_lat_queue) # Use real sparsity to estimate lat
      else:
        estimated_time = sum(self.running_task[task_id].est_lat_queue) # Use estimate avg (PREMA) lat  
      slack_time =  task.target_time - self.sys_time
      torun_lat = sum(task.real_lat_queue)
      logging.debug("task in candidate:%s with estimated time:%f, target time:%f, slack time:%f, torun time:%f" % (task_id, estimated_time, task.target_time, slack_time, torun_lat))
      if ((next_task_id is None) or (shortest_time > estimated_time)): 
        next_task_id = task_id
        shortest_time = estimated_time
    logging.debug("next task:%s, sys time:%f" % (next_task_id, self.sys_time))
    return next_task_id