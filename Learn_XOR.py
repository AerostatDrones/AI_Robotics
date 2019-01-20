

from microMLP import MicroMLP

mlp = MicroMLP.Create( neuronsByLayers           = [4, 6, 2],
                       activationFuncName        = MicroMLP.ACTFUNC_TANH,
                       layersAutoConnectFunction = MicroMLP.LayersFullConnect )

nnFalse  = MicroMLP.NNValue.FromBool(False)
nnTrue   = MicroMLP.NNValue.FromBool(True)

mlp.AddExample( [nnFalse, nnFalse, nnFalse, nnFalse], [nnFalse, nnFalse] )
mlp.AddExample( [nnFalse, nnTrue, nnTrue, nnFalse  ], [nnTrue, nnTrue  ] )
mlp.AddExample( [nnTrue , nnTrue, nnFalse, nnFalse ], [nnTrue, nnFalse ] )
mlp.AddExample( [nnFalse, nnFalse, nnTrue, nnTrue  ], [nnFalse, nnTrue ] )
mlp.AddExample( [nnFalse, nnTrue, nnFalse, nnFalse ], [nnTrue,  nnFalse] )
mlp.AddExample( [nnFalse, nnFalse, nnTrue, nnFalse ], [nnFalse, nnTrue ] )
mlp.AddExample( [nnTrue, nnFalse, nnFalse, nnFalse ], [nnTrue, nnFalse ] )
mlp.AddExample( [nnFalse, nnFalse, nnFalse, nnTrue ], [nnFalse, nnTrue ] )
mlp.AddExample( [nnTrue, nnTrue, nnTrue, nnTrue  ],   [nnFalse, nnFalse] )

learnCount = mlp.LearnExamples(maxSeconds= 0xffffffffffffffff)

print( "LEARNED :" )
print( "  - False xor False = %s" % mlp.Predict([nnFalse, nnFalse, nnTrue, nnTrue])[0].AsBool)
print( "  - False xor False = %s" % mlp.Predict([nnFalse, nnFalse, nnTrue, nnTrue])[1].AsBool)

#print( "  - False xor True  = %s" % mlp.Predict([nnFalse, nnTrue] )[0].AsBool )
#print( "  - True  xor True  = %s" % mlp.Predict([nnTrue , nnTrue] )[0].AsBool )
#print( "  - True  xor False = %s" % mlp.Predict([nnTrue , nnFalse])[0].AsBool )

if mlp.SaveToFile("mlp_line.json") :
	print( "MicroMLP structure saved!" )





