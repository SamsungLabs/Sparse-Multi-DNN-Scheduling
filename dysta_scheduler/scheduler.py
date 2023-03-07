#Remeber to set random seed
from utils import *

import logging
logger = logging.getLogger(__name__)

class Scheduler:
  def __init__(self, reqst_table):
    self.sys_time = 0.0
    self.reqst_table = reqst_table # [(req_time1, target_lat1, model1), (req_time2, target_lat2, model2), ....]
    self.cur_reqst_indx = 0
    self.running_task = {} #{{task_id1: task1}, {task_id2: task2} ...}
    self.finished_reqst = {}
    self.num_reqst= len(self.reqst_table)
    self.lat_lut = {}

  def is_finish(self):
    return len(self.finished_reqst) == self.num_reqst

  def set_lat_lut(self, lat_lut):
    self.lat_lut = lat_lut

  def cal_violation_rate(self):
    assert self.is_finish() # Check if all finished
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

  def __push_reqst(self, reqst_indx):
    reqst_time, target_lat, reqst_model, priority = self.reqst_table[reqst_indx]
    # Construct new task
    new_task = Task(reqst_time, target_lat, reqst_model, priority)
    num_examples = len(self.lat_lut[reqst_model]) # Get datset
    sample_id = new_task.sample_data(num_examples) # Sample from data
    new_task.construct_task(self.lat_lut[reqst_model]['lat_lut'][sample_id])
    task_id = str(reqst_time) + "_" + reqst_model
    # Add task into running tasks
    assert task_id not in self.running_task
    # print ("push reqst:", task_id)
    self.running_task[task_id] = new_task

  def update_reqst(self):
    if (self.cur_reqst_indx < self.num_reqst): 
      # request info: req_time, target_lat, model, priority
      reqst_time, target_lat, reqst_model, priority = self.reqst_table[self.cur_reqst_indx]
      if (len(self.running_task) == 0): # No running task, push the nearst request
        self.__push_reqst(self.cur_reqst_indx)
        self.cur_reqst_indx += 1
        self.sys_time = reqst_time
      else: # Push all the request behind the sys_time
        while (reqst_time < self.sys_time):
          self.__push_reqst(self.cur_reqst_indx)
          # Update request info
          self.cur_reqst_indx += 1
          if (self.cur_reqst_indx >= self.num_reqst): break
          reqst_time, target_lat, reqst_model, priority = self.reqst_table[self.cur_reqst_indx]
    else:
      pass
  
  # Virtual class
  def update_schedule(self):
    raise NotImplementedError()

  def exe(self): # Per layer granlarity
    task_id = self.update_schedule() # Return exe task id
    lat_consumed = self.running_task[task_id].exe()
    self.sys_time += lat_consumed
    self.running_task[task_id].prema_last_exe_time = self.sys_time
    # Check any finished task
    if (self.running_task[task_id].is_finish(self.sys_time)):
      self.finished_reqst[task_id] = self.running_task[task_id]
      del self.running_task[task_id]
      # print ("finish reqst:", task_id)




class FCFS_Scheduler(Scheduler):
  def __init__(self,reqst_table):
    super().__init__(reqst_table)
    print ("Constructing FCFS Scheduler")

  def reset(self, reqst_table):
    self.__init__(reqst_table)

  def update_schedule(self):
    next_task_id = None
    nearst_reqst_time = -1
    for k,v in self.running_task.items():
      if ((nearst_reqst_time < 0) or (nearst_reqst_time > v.reqst_time)): 
        next_task_id = k
        nearst_reqst_time = v.reqst_time

    logging.debug("next task:%s, sys time:%f" % (next_task_id, self.sys_time))
    return next_task_id
    # raise NotImplementedError()



class PREAMA_Scheduler(Scheduler):
  def __init__(self,reqst_table):
    super().__init__(reqst_table)
    self.ready_queue = {}

  def reset(self, reqst_table):
    self.__init__(reqst_table)

  def update_schedule(self):
    next_task_id = None
    # Update token score
    max_preama_token = 0
    max_task_id = None
    for k,v in self.running_task.items():
      if (v.preama_token < 0): 
        v.preama_token = v.priority # Line 3 in PREMA paper, Algorithm 2
        max_task_id = k
      else: 
        # Line 7 in PREMA paper, Algorithm 2
        idle_time =  self.sys_time - v.prema_last_exe_time
        slowdown_rate = idle_time / v.isolated_time
        v.preama_token += v.priority * slowdown_rate
        if max_preama_token < v.preama_token:
          max_preama_token = v.preama_token
          max_task_id = k
    # Get threshold based on the max threshould, according to the paragragh under TABLE II in the PREMA paper
    threshold = -1
    if (max_preama_token >= 9): threshold = 9
    elif (max_preama_token >= 3): threshold = 3
    else: threshold = 1
    # Get candidate, Line 10 in PREMA paper, Algorithm 2. 
    candidates = []
    for k,v in self.running_task.items():
      ###################!!!!!!!We change it to "Token >= Threshold", or it does not make sense at the beginning!!!!!!!!!###########
      if (v.preama_token >= threshold): candidates.append(k)
    # Get next token using shortest estimated time approach
    logging.debug("threshold:%d" % (threshold))
    shorted_time = -1
    for k in candidates:
      estimated_time = sum(self.running_task[k].lat_queue)
      if len(candidates) > 1:
        logging.debug("task in candidate:%s with estimated time:%f" % (k, estimated_time))
      #if ((next_task_id is None) or (self.running_task[next_task_id].isolated_time > self.running_task[k].isolated_time)): 
      if ((next_task_id is None) or (shorted_time > estimated_time)): 
        next_task_id = k
        shorted_time = estimated_time
    logging.debug("next task:%s, sys time:%f" % (next_task_id, self.sys_time))
    return next_task_id
    # logging.debug("next task:%s, sys time:%f" % (next_task_id, self.sys_time))




class Dysta_Scheduler(Scheduler):
  def __init__(self,reqst_table):
    super().__init__(reqst_table)
    print ("Constructing Dysta Scheduler")
  def reset(self, reqst_table):
    self.__init__(reqst_table)

  def update_schedule(self):
    next_task_id = None
    highest_urgency = -1
    for k,v in self.running_task.items():
      remain_lat =  v.target_time - self.sys_time
      torun_lat = sum(v.lat_queue)
      v.urgency = torun_lat/remain_lat
      v.urgency = 1 if v.urgency > 1 else v.urgency
      print ("cur urgency:", v.urgency, " and highest urgency:", highest_urgency)
      if ((highest_urgency < 0) or (highest_urgency <= v.urgency)): 
        if ((highest_urgency == v.urgency) and (self.running_task[next_task_id].reqst_time < v.reqst_time)):
          # If with the same, use FCFS
          continue
        else:
          next_task_id = k
          highest_urgency = v.urgency
    logging.debug("next task:%s, sys time:%f" % (next_task_id, self.sys_time))
    return next_task_id


