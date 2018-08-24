import gym, math, random
import numpy as np
#env = gym.make('CartPole-v0')
env = gym.make('MountainCar-v0')
#env = gym.make('Hopper-v1')
#env = gym.make('MsPacman-v0')


A_DIVS=100
B_DIVS=100


aRange = ( -2, 2 ) #
bRange = ( -.1, .1 ) #


brain = np.zeros( (A_DIVS,B_DIVS, 3) )
#brain += 4.5

obs = env.reset()

LEFT = 0
NOTHING = 1
RIGHT = 2

discount = .99999

running_average_value = 0

def obs_to_index( _obs ):
    result = [ int((_obs[0]-aRange[0])/(aRange[1]-aRange[0])*A_DIVS + .5  ),
             int((_obs[1]-bRange[0])/(bRange[1]-bRange[0])*B_DIVS + .5  ),]

    result[0] = max(0,min(A_DIVS-1,result[0]))
    result[1] = max(0,min(B_DIVS-1,result[1]))


    return result

steps_till_crash = 0
run_number = 0


count_out = 0

while True:

    
    epsilon = 1.0/((run_number+1)/1)

    indexes = obs_to_index( obs )

    left_value = brain[ indexes[0], indexes[1], LEFT ]
    no_value   = brain[ indexes[0], indexes[1], NOTHING ]
    right_value = brain[ indexes[0], indexes[1], RIGHT ]


    if random.random() > epsilon:
        best_value = left_value
        picked_direction = LEFT
        if no_value > best_value:
            picked_direction = NOTHING
            best_value = no_value
        if right_value > best_value:
            picked_direction = RIGHT
            best_value = right_value

    else:
        picked_direction = random.choice( [LEFT,NOTHING,RIGHT])

    #if picked_direction == LEFT:
   #     print( "vl:" + str(left_value), end=' ' )
   # else:
    #    print( "vr:" + str(right_value), end=' ' )

    results = env.step(picked_direction)

    obs = results[0]
    reward = results[1]
    done = results[2]

    new_index = obs_to_index( obs )

    thing = max(max( brain[ new_index[0], new_index[1], LEFT], brain[new_index[0], new_index[1], RIGHT] ), brain[new_index[0], new_index[1], NOTHING])
    
    if done:
        target_value = reward
        #target_value = -.5 #-5
        

        #alpha = .99
        #running_average_value = alpha*running_average_value + (1-alpha)*steps_till_crash

        #print( "epsilon at : " + str( epsilon ) + " r" + str(run_number) + " stepped " + str(steps_till_crash) + " till crash or " + str( running_average_value ) )
        #steps_till_crash = 0
    else:
        target_value = thing*discount + reward
        #steps_till_crash += 1


    existing_value = brain[ indexes[0], indexes[1], picked_direction ]

    brain[ indexes[0], indexes[1], picked_direction ] = existing_value * .9 + target_value * .1
    #print( "changing from " + str( existing_value ) + " closer to " + str( target_value ) + " indexing at " + str( indexes[0] ) + ", " + str( indexes[1] ) )

    #if existing_value * .9 + target_value * .1 > 1:
    #    print( "ahahahah" )

    #steps_final = steps_till_crash_array[-1]

    if run_number % 2000 == 0:
    #if steps_final == steps_till_crash_max:
        env.render()

        
    if done:
        env.reset()
        run_number += 1
        print( "now on run " + str( run_number ) )

    #env.render()
    
    

        

    
