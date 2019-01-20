from microMLP import MicroMLP

mlp = MicroMLP.Create( neuronsByLayers           = [2, 3, 2],
                       activationFuncName        = MicroMLP.ACTFUNC_SIGMOID,
                       layersAutoConnectFunction = MicroMLP.LayersFullConnect )

false  = MicroMLP.NNValue(0,1,0) #(Min , Max, Value)
true   = MicroMLP.NNValue(0,1,1)

motor_right = mlp.NNValue(0,100,0) # 0 for CW Rotation
motor_left  = mlp.NNValue(0,100,100) # 100 for CCW Rotation
stop = mlp.NNValue.FromPercent(50.7) # 50 stops servo Rotation

mlp.AddExample( [false, false], [motor_right, motor_left] ) # ([Input] , [OutPut]) as vector values
mlp.AddExample( [false, true ], [motor_right, stop ] )
mlp.AddExample( [true , true], [stop, stop] )
mlp.AddExample( [true , false], [stop, motor_left] )

learnCount = mlp.LearnExamples(maxSeconds= 0xffffffffffffffff) # Max time to learn, default 30 seconds

print( "LEARNED :" )
print( "  - False and False = %s , %s" , mlp.Predict([false, false])[0].AsPercent, mlp.Predict([false, false])[1].AsPercent)
print( "  - False and True  = %s , %s" , mlp.Predict([false, true] )[0].AsPercent,  mlp.Predict([false, true] )[1].AsPercent )
print( "  - True  and True  = %s , %s" , mlp.Predict([true , true] )[0].AsPercent,  mlp.Predict([true , true] )[1].AsPercent )
print( "  - True  and False = %s , %s" , mlp.Predict([true , false])[0].AsPercent,  mlp.Predict([true , false])[1].AsPercent )

if mlp.SaveToFile("line_follower.json") :
	print( "MicroMLP structure saved!" )




