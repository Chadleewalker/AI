import gym, math, random
import numpy as np
#env = gym.make('CartPole-v0')
env = gym.make('MountainCar-v0')
#env = gym.make('Hopper-v1')
#env = gym.make('MsPacman-v0')
import tensorflow as tf

batch_size = 200
TRAIN_STEPS = 1000

#custom estimator page
#https://www.tensorflow.org/guide/custom_estimators

#custom estimator example
#https://github.com/tensorflow/models/blob/master/samples/core/get_started/custom_estimator.py

#esitmator source code
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/estimators/estimator.py

def my_model( features, labels, mode, params ):
    #net = tf.feature_column.input_layer(features, params['feature_columns'])
    #net = tf.transpose(features)
    net = tf.expand_dims(features, 0)

    print( "net is " )
    print( net )

    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)


    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)


    # return predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    print( "labels" )
    print( labels )

    print( "logits" ) 
    print( logits )
    #print( logits.shape() )

     # Compute loss
    logits = tf.reshape( logits, [] )
    loss = tf.losses.mean_squared_error( labels, logits )

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss )

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


the_new_brain = tf.estimator.Estimator(
    model_fn=my_model,
    params={
            #'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 3 classes.
            'n_classes': 1,
        })



observations_x = [
    
]
observations_y = [

]


def input_function():
    dataset = tf.data.Dataset.from_tensor_slices( (observations_x, observations_y) )
    dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset

have_thought = [False]

def think_about_life():
    print( "Training!!!" )
    the_new_brain.train(
        input_fn=input_function,
        steps=TRAIN_STEPS)
    have_thought[0] = True


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

#def obs_to_index( _obs ):
#    result = [ int((_obs[0]-aRange[0])/(aRange[1]-aRange[0])*A_DIVS + .5  ),
#             int((_obs[1]-bRange[0])/(bRange[1]-bRange[0])*B_DIVS + .5  ),]
#
#    result[0] = max(0,min(A_DIVS-1,result[0]))
#    result[1] = max(0,min(B_DIVS-1,result[1]))
#
#
#    return result

steps_till_crash = 0
run_number = 0


count_out = 0

while True:

    
    epsilon = 1.0/((run_number+1)/1)

    left_value = random.random()
    no_value = random.random()
    right_value = random.random()
    try:
        if have_thought[0]:
            left_predictor = the_new_brain.predict( lambda: [( LEFT   , obs[0], obs[1])] )
            #left_predictor = left_predictor()
            left_value  = next(left_predictor)["logits"][0]
            no_value    = next(the_new_brain.predict( lambda: [( NOTHING, obs[0], obs[1])] ))["logits"][0]
            right_value = next(the_new_brain.predict( lambda: [( RIGHT  , obs[0], obs[1])] ))["logits"][0]
    except ValueError as ex:
        pass
        #print( "problem over here: " + str( ex ) )



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

    #new_index = obs_to_index( obs )

    #C:\\Users\\chadl\\AppData\\Local\\Temp\\tmpz3c1w4nk

    #thing = max(max( brain[ new_index[0], new_index[1], LEFT], brain[new_index[0], new_index[1], RIGHT] ), brain[new_index[0], new_index[1], NOTHING])
    new_left_value = random.random()
    new_no_value = random.random()
    new_right_value = random.random()
    try:
        new_left_value  = next(the_new_brain.predict( lambda: [( LEFT   , obs[0], obs[1])] ))["logits"][0]
        new_no_value    = next(the_new_brain.predict( lambda: [( NOTHING, obs[0], obs[1])] ))["logits"][0]
        new_right_value = next(the_new_brain.predict( lambda: [( RIGHT  , obs[0], obs[1])] ))["logits"][0]
    except ValueError as ex:
        pass
        #print( "somesing not hapey. i.e. sad potato: " + str(ex) )

    thing = max( new_left_value, new_no_value, new_right_value )
    
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


    existing_value = 0
    try:
        existing_value = next(the_new_brain.predict( lambda: [(picked_direction, obs[0], obs[1] )] ))["logits"][0]
    except ValueError as bob:
        pass
        #print( "bob isn't happy: " + str( bob ) )

    
    next_value = existing_value * .9 + target_value * .1

    observations_x.append( (picked_direction, obs[0], obs[1] ) )
    observations_y.append( next_value )

    #print( "changing from " + str( existing_value ) + " closer to " + str( target_value ) + " indexing at " + str( indexes[0] ) + ", " + str( indexes[1] ) )

    #if existing_value * .9 + target_value * .1 > 1:
    #    print( "ahahahah" )

    #steps_final = steps_till_crash_array[-1]

    if run_number % 2000 == 0 and run_number > 0:
    #if steps_final == steps_till_crash_max:
        env.render()

        
    if done:
        env.reset()
        run_number += 1
        print( "now on run " + str( run_number ) )

        if run_number % 200 == 0:
            think_about_life()
            observations_x = []
            observations_y = []

    #env.render()
    
    

        

    
