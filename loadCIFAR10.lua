require 'CIFAR10'
function loadCIFAR10(fileDIR) 

	local trainData, testData = {},{}
	
	trainData, testData  = CIFAR10()
	
--	trainData = torch.load(fileDIR .. "/CIFAR10_trainData.t7")
--	testData  = torch.load(fileDIR .. "/CIFAR10_testData.t7")	
  
  
  
	print('Training Data:')
	print(trainData)
	print()

--	print('Validation Data:')
--	print(validData)
--	print()
  
	print('Testing Data:')
	print(testData)
	print()		

	return trainData, testData

end
