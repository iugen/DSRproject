import os
import sys
import numpy as np
import traci
from Agent import agent_class

		

def TL_state(detectors_ID):	
	state = []		
	for detector in detectors_ID:
		speed = traci.lanearea.getLastStepMeanSpeed(detector)
		# Returns the mean speed of vehicles that were 
		# on the named induction loop within the last simulation step [m/s]
		state.append(speed)

	state = np.array(state)
	state = state.reshape((1, state.shape[0]))
	return state



def TL_reward(s):
	reward = s.mean()
	return reward


def run():
	traci.start(sumoCmd) # starts the SUMO simulation defined in sumoCmd 

	TL_id = traci.trafficlight.getIDList() # list of traffic lights (just one n this case)
	total_step = 2000		# total step of the simulation
	detectors_ID = traci.lanearea.getIDList()
	n_epochs = 5
	agent = agent_class()
	traci.close()

	for i in range(n_epochs):
		traci.start(sumoCmd)
		state = TL_state(detectors_ID)  
		total_reward = 0
		step = 0
		number_of_phases = 6
		int_step = 0

		while step < total_step:
			if step % 10 == 0:     # udate the state every 10 seconds
				action = agent.select_action(state)  # 0 or 1 (stay or change - simple control logic)

				curr_phase = traci.trafficlight.getPhase(TL_id[0])
				next_phase = (curr_phase + action) % number_of_phases
				traci.trafficlight.setPhase(TL_id[0], next_phase)
				traci.simulationStep()		# executes one step of the simulation (= 1 sec)
				next_s = TL_state(detectors_ID)
				reward = TL_reward(next_s)
				agent.memory(state,reward,next_s,action) # saves state and actions as data to train the neural network
				total_reward += reward
				state = next_s
			else:
				traci.simulationStep()

			step += 1

		traci.close()
		agent.fit_NN()		# fit the NN model



sumoBinary = "C:/Program Files (x86)/DLR/Sumo/bin/sumo" # SUMO path
sumoCmd = [sumoBinary, "-c", "SUMO_files/cross.sumocfg"]	
# the .cfg file contains the information for the SUMO simulation
run()


'''    phases of the traffic light
<tlLogic id="c" type="static" programID="0" offset="0">
    <phase duration="400" state="GGGgrrrrGGGgrrrr"/>
    <phase duration="5" state="yyyyrrrryyyyrrrr"/>
    <phase duration="2" state="rrrrrrrrrrrrrrrr"/>
    <phase duration="400" state="rrrrGGGgrrrrGGGg"/>
    <phase duration="5" state="rrrryyyyrrrryyyy"/>
    <phase duration="2" state="rrrrrrrrrrrrrrrr"/>
</tlLogic>   '''