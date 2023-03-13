from utils import Task
import logging
import numpy as np

class Scheduler:
  def __init__(self, reqst_table):
    self.sys_time = 0.0
    self.reqst_table = reqst_table # [(req_time1, target_lat1, model1), (req_time2, target_lat2, model2), ....]
    self.cur_reqst_indx = 0
    self.running_task = {} #{{task_id1: task1}, {task_id2: task2} ...}
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
    violate_task_list = [] #[(reqst, target, model_str), ....]
    for task_id, task_info in self.finished_reqst.items():
      if task_info.finish_time > task_info.target_time:
        num_violate_tasks += 1
        violate_task_list.append({"reqst_time": task_info.reqst_time, "target_time:": task_info.target_time, 
                      "finsh_time:": task_info.finish_time, "task_id": task_id})
    violation_rate = num_violate_tasks/self.num_reqst
    assert (violation_rate >= 0) and (violation_rate <= 1)
    return violation_rate, violate_task_list

  def calc_system_thrpt(self):
    """
    Returns the system throughput (STP) in inferences per second (inf/s).
    """
    assert self.is_finished()
    last_finish_time = 0
    for task_id,task_info in self.finished_reqst.items():
      if last_finish_time <= task_info.finish_time:
        last_finish_time = task_info.finish_time
    first_reqst_time, _, _, _, _, _ = self.reqst_table[0]
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
    reqst_time, target_lat, reqst_model, priority, avg_lat, sample_id = self.reqst_table[reqst_indx]
    
    # Construct a new task
    new_task = Task(reqst_time, target_lat, reqst_model, priority, avg_lat)


    new_task.construct_task(self.lat_lut[reqst_model]['lat_lut'][sample_id])
    task_id = str(reqst_time) + "_" + reqst_model # log using arrival time and model name
    
    # Add task into running tasks
    assert task_id not in self.running_task
    # print ("push reqst:", task_id)
    self.running_task[task_id] = new_task

  def update_reqst(self):
    if (self.cur_reqst_indx < self.num_reqst): 
      # request info: req_time, target_lat, model, priority
      reqst_time, target_lat, reqst_model, priority, avg_lat, sample_id = self.reqst_table[self.cur_reqst_indx]
      
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
          reqst_time, target_lat, reqst_model, priority, avg_lat, sample_id = self.reqst_table[self.cur_reqst_indx]
    else:
      pass
  
  # Virtual class - to be implemented by each scheduler
  def update_schedule(self):
    raise NotImplementedError()

  def exe(self): # Per-layer granularity
    task_id = self.update_schedule() # id of the scheduled task
    lat_consumed = self.running_task[task_id].exe() # latency of the executed layer
    self.sys_time += lat_consumed
    self.running_task[task_id].prema_last_exe_time = self.sys_time
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
        idle_time =  self.sys_time - task.prema_last_exe_time
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
        estimated_time = sum(self.running_task[task_id].prema_est_lat_queue) # Use estimate avg (PREAMA) lat  
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
  This scheduler implements our proposed approach.
  """
  def __init__(self,reqst_table):
    super().__init__(reqst_table)
    print ("Constructing Dysta Scheduler.")
  def reset(self, reqst_table):
    self.__init__(reqst_table)

  def update_schedule(self):
    next_task_id = None
    highest_urgency = -1
    for task_id,task in self.running_task.items():
      remain_lat =  task.target_time - self.sys_time
      torun_lat = sum(task.real_lat_queue)
      task.urgency = torun_lat/remain_lat
      task.urgency = 1 if task.urgency > 1 else task.urgency
      if ((highest_urgency < 0) or (highest_urgency <= task.urgency)): 
        if ((highest_urgency == task.urgency) and (self.running_task[next_task_id].reqst_time < task.reqst_time)):
          # If with the same, use FCFS
          continue
        else:
          next_task_id = task_id
          highest_urgency = task.urgency
    logging.debug("next task:%s, sys time:%f" % (next_task_id, self.sys_time))
    return next_task_id


