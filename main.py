import os
import sys
import numpy as np
import traci
from Agent import AgentClass

		

def TL_state(detectors_ID):	
	state = []		
	for detector in detectors_ID:
		speed = traci.lanearea.getLastStepMeanSpeed(detector)
		# Returns the mean speed of vehicles that were 
		# on the named induction loop within the last simulation step [m/s]
		state.append(speed)

		n_veh = traci.lanearea.getLastStepVehicleNumber(detector)
		# Returns the mean speed of vehicles that were 
		# on the named induction loop within the last simulation step [m/s]
		state.append(n_veh)

	state.append(traci.trafficlight.getPhase(traci.trafficlight.getIDList()[0]))
		

	state = np.array(state,)
	state = state.reshape((1, state.shape[0]))
	return state


def TL_reward(detectors_ID,previous_reward):
	reward_sum = 0
	for detector in detectors_ID:
		reward = -traci.lanearea.getJamLengthVehicle(detector)
		reward_sum += reward   
	#reward_sum += previous_reward
	return reward_sum

	# reward_sum: ne passano solo due alla volta
	# + phase: loss is nan
	# += previous_reward: non cambia mai
	# + phase: loss is nan<
	# altre function?
	# tolto primo if dopo while (faccio il training ogni secondo, invece che dieci)


def run():
	traci.start(sumoCmd) # starts the SUMO simulation defined in sumoCmd 

	TL_id = traci.trafficlight.getIDList() # list of traffic lights (just one n this case)
	total_step = 2000		# total step of the simulation
	detectors_ID = traci.lanearea.getIDList()
	n_episodes = 100
	update_time = 10
	agent = AgentClass(1)
	traci.close()

	for i in range(n_episodes):
		traci.start(sumoCmd)
		state = TL_state(detectors_ID)  
		total_reward = 0
		previous_reward = 0
		step = 1
		number_of_phases = 6    # TO DO: get it from traci
		done = 1

		while step <= total_step:
			if step % update_time == 0:     # udate the state every 10 seconds
				action = agent.select_action(state)  # 0 or 1 (stay or change - simple control logic)
				# print('action: ', action)
				curr_phase = traci.trafficlight.getPhase(TL_id[0])
				next_phase = (curr_phase + action) % number_of_phases
				traci.trafficlight.setPhase(TL_id[0], next_phase)
				traci.simulationStep()		# executes one step of the simulation (= 1 sec)
				next_s = TL_state(detectors_ID)
				reward = TL_reward(detectors_ID, previous_reward)
				if step == total_step - update_time:
					done = 0
				agent.memory(state,reward,next_s,action,done) # saves state and actions as data to train the neural network
				total_reward += reward
				state = next_s
				previous_reward = reward
				if i > 0:
					agent.fit_NN()		# fit the NN model, starting from epoch 2

			else:
				traci.simulationStep()


			step += 1

		agent.update_target_network()
		traci.close()

	agent.save('model_weights.h5')



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