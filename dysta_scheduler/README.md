# Simulation of scheduling

Pls install the environment as instructed in previous directory.

## Run on Sanger

1. Download the latency csv file from the link [here](https://drive.google.com/file/d/1r6daW3wEzgyMQk-_ufu961ZHPhj0Epdk/view?usp=sharing), put it under this folder
2. Run the following command to compare fcfs with prema
```
python simulator.py --schedule_method fcfs prema
# you can also enable the debug info
python simulator.py --schedule_method fcfs prema --debug 
```