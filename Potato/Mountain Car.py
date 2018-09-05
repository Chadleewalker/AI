import gym, math, random
import numpy as np
#env = gym.make('CartPole-v0')
env = gym.make('MountainCar-v0')
#env = gym.make('Hopper-v1')
#env = gym.make('MsPacman-v0')

steps_till_crash_array = [0]
steps_till_crash_max = 0
r_at__199 = 0
i = 0

A_DIVS=10
B_DIVS=10
C_DIVS=10
D_DIVS=10


aRange = ( -4.5, 4.5 ) #
bRange = ( -4.5, 4.5 ) #
cRange = ( -1.5, 1.5 ) #
dRange = ( -1.5, 1.5 ) #

brain = np.zeros( (A_DIVS,B_DIVS, C_DIVS, D_DIVS, 2 ) )
brain += 4.5

obs = env.reset()

LEFT = 0
RIGHT = 1

discount = .99999

running_average_value = 0

def obs_to_index( _obs ):
    result = [ int((_obs[0]-aRange[0])/(aRange[1]-aRange[0])*A_DIVS + .5  ),
             int((_obs[1]-bRange[0])/(bRange[1]-bRange[0])*B_DIVS + .5  ),
             int((_obs[2]-cRange[0])/(cRange[1]-cRange[0])*C_DIVS + .5  ),
             int((_obs[3]-dRange[0])/(dRange[1]-dRange[0])*D_DIVS + .5  ) ]

    result[0] = max(0,min(A_DIVS-1,result[0]))
    result[1] = max(0,min(B_DIVS-1,result[1]))
    result[2] = max(0,min(C_DIVS-1,result[2]))
    result[3] = max(0,min(D_DIVS-1,result[3]))

    return result

steps_till_crash = 0
run_number = 0

while True:

    
    epsilon = 1.0/((run_number+1)/1)

    indexes = obs_to_index( obs )

    left_value = brain[ indexes[0], indexes[1],indexes[2],indexes[3], LEFT ]
    right_value = brain[ indexes[0], indexes[1],indexes[2], indexes[3], RIGHT ]


    if random.random() > epsilon:
        picked_direction = LEFT
        if right_value > left_value:
            picked_direction = RIGHT
    else:
        picked_direction = LEFT if random.random() > .5 else RIGHT

    #if picked_direction == LEFT:
   #     print( "vl:" + str(left_value), end=' ' )
   # else:
    #    print( "vr:" + str(right_value), end=' ' )

    results = env.step(picked_direction)

    obs = results[0]
    reward = results[1]
    done = reward == 0 #results[2]

    if reward == 0:
        reward = -10000

    new_index = obs_to_index( obs )

    thing = max( brain[ new_index[0], new_index[1],new_index[2],new_index[3], LEFT], brain[new_index[0], new_index[1], new_index[2], new_index[3], RIGHT] )

    if steps_till_crash == 199:
        print( "I am a potato!" )
    
    if done:
        target_value = reward
        #target_value = -10000 #-5
        #steps_till_crash_array.append(steps_till_crash)
        steps_till_crash_max = max(steps_till_crash_array)
        if steps_till_crash == 199 and i == 0:
            env.render()
            r_at__199 = run_number
            i += 1

        alpha = .99
        running_average_value = alpha*running_average_value + (1-alpha)*steps_till_crash

        print( "epsilon at : " + str( epsilon ) + " r" + str(run_number) + " stepped " + str(steps_till_crash) + " till crash or " + str( running_average_value ) )
        steps_till_crash = 0
    else:
        target_value = thing*discount + reward
        steps_till_crash += 1


    existing_value = brain[ indexes[0], indexes[1],indexes[2], indexes[3], picked_direction ]

    brain[ indexes[0], indexes[1],indexes[2], indexes[3], picked_direction ] = existing_value * .9 + target_value * .1

    #if existing_value * .9 + target_value * .1 > 1:
    #    print( "ahahahah" )
    steps_final = steps_till_crash_array[-1]

    if run_number % 2000 == 0:
    #if steps_final == steps_till_crash_max:
        env.render()

        
    if done:
        env.reset()
        run_number += 1

    #env.render()
    
    

        

    
