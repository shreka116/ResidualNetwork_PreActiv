require 'loadBottleNeckModel'
require 'loadOptimizer'
require 'cudnn'
require 'confusionMatrix'

function trainBottleNeck_new(trainData,testData,numLayer)
	------------------------------------
	-- define models
	------------------------------------
  print('Training Residual Network ....')

  local subBatchSize = 64
  local model = loadBottleNeckModel(numLayer)
  model:cuda()
  cudnn.convert(model,cudnn)
  print(model)
  
  local lossFunc  = nn.ClassNLLCriterion()
  lossFunc:cuda()
   
  local opt, optState, optMethod = loadOptimizer('164')
  
  
  local params, grads = model:getParameters()
  local batch_counter = 0
  
  local randPerm = torch.randperm(trainData:size(1)):long()
  local randRow, randCol = {},{}

--  local batchData = torch.Tensor(opt.batchSize, 3, 32, 32):fill(0)
--  local batchLabel= torch.Tensor(opt.batchSize) 

  local paddedData= torch.Tensor(1,3,40,40):fill(0)  
  local tmpData   = torch.Tensor(1,3,32,32):fill(0) 
 
  local batch_start_idx = 1

 local feval = function(params_new)
      
    if params ~= params_new then
      params:copy(params_new)
    end
    
--    randPerm        = torch.randperm(trainData:size(1)):long()
--   local batchData = torch.Tensor(math.min(subBatchSize, trainData.data:size(1)-batch_start_idx+1),3,32,32):fill(0)
--   local batchLabel= torch.Tensor(math.min(subBatchSize, trainData.data:size(1)-batch_start_idx+1))
    local batchData = torch.zeros(math.min(opt.batchSize, trainData.data:size(1)-batch_start_idx+1), 3, 32, 32)
    local batchLabel= torch.zeros(math.min(opt.batchSize, trainData.data:size(1)-batch_start_idx+1))     
    
    batchData[{ {1,math.min(opt.batchSize, trainData.data:size(1)-batch_start_idx+1)},{},{},{} }]       = trainData.data:index(1,randPerm[{ {batch_start_idx, math.min(batch_start_idx+opt.batchSize-1, trainData.data:size(1))} }])
--    batchData[{ {opt.batchSize/2 + 1, opt.batchSize},{},{},{} }] = image.hflip(trainData.data:index(1, randPerm[{ {opt.batchSize/2 + 1, opt.batchSize} }]))
    
    batchLabel[{ {1,math.min(opt.batchSize, trainData.data:size(1)-batch_start_idx+1)} }] = trainData.labels:index(1, randPerm[{ {batch_start_idx, math.min(batch_start_idx + opt.batchSize - 1, trainData.data:size(1))} }])
--    batchLabel[{ {opt.batchSize/2+1, opt.batchSize} }] = trainData.labels:index(1, randPerm[{ {opt.batchSize/2+1, opt.batchSize} }])



    for iter = 1, batchData:size(1) do
  	--local paddedData= torch.Tensor(1,3,40,40):fill(0)  
	randRow = torch.ceil(torch.rand(1)*9)[1]
        randCol = torch.ceil(torch.rand(1)*9)[1]
        paddedData[{ {},{},{5,36},{5,36} }] = batchData[{ {iter},{},{},{} }]:contiguous()
	if torch.rand(1)[1] > 0.5 then
	  batchData[{ {iter},{},{},{} }] = image.hflip(paddedData[{ {},{},{randRow,randRow+31},{randCol,randCol+31} }]:reshape(3,32,32)):reshape(1,3,32,32)
	else
    	  batchData[{ {iter},{},{},{} }] = paddedData[{ {},{},{randRow,randRow+31},{randCol,randCol+31} }]
  	end

    end

    model:training()    
--    grads:zero()
    local acc_err = 0
    local acc_grads = torch.zeros(grads:size()):cuda()
    local batchIterator = math.min(opt.batchSize, trainData.data:size(1) - batch_start_idx + 1)/subBatchSize
    for batchIter = 1, batchIterator do
      grads:zero()
     
      local subBatchData = batchData[{ {1 + (batchIter-1)*subBatchSize, math.min(batchIter*subBatchSize, batchData:size(1))},{},{},{} }]
      local subBatchLabel= batchLabel[{ {1 + (batchIter-1)*subBatchSize, math.min(batchIter*subBatchSize, batchData:size(1))} }]
 --    print(subBatchData:size())
 --    print(subBatchLabel:size()) 
      --feedforward
      local trainOutputs = model:forward(subBatchData:cuda())
      --compute loss
      local trainLoss    = lossFunc:forward(trainOutputs, subBatchLabel:cuda())
--      local err = trainLoss*(math.min(subBatchSize, opt.batchSize - (batchIter-1)*subBatchSize))
      --compute the derivative of the loss
      local dLoss       = lossFunc:backward(trainOutputs, subBatchLabel:cuda())
--      local tmpLoss 	= dLoss*(math.min(subBatchSize, opt.batchSize - (batchIter-1)*subBatchSize))
--	print(dLoss)
--	print(tmpLoss)
      model:backward(subBatchData:cuda(), dLoss)
      acc_err = acc_err + trainLoss

--	print(batchIter)
--	print(grads:type())
--	print(grads:size())
--	print(acc_grads:type())

      acc_grads:add(grads:cuda())
    end
--    trainLoss = acc_err/opt.batchSize
--    grads:div(opt.batchSize)
    trainLoss = acc_err/batchIterator
    acc_grads:div(batchIterator)
  
    return trainLoss, acc_grads
      
  end
  
  --------------------------
  -- optimization loop
  --------------------------
  local losses = {}
  local numIter = opt.maxIter
  local trError = {}
--  local vlError = {}
  local tsError = {}
  local rndIDX = {}
    
  local tmp = 0
 

  for iter = 1, numIter do
    if iter == 32000 then
	optState.learningRate = optState.learningRate/10
    elseif iter == 48000 then
    	optState.learningRate = optState.learningRate/10
    end
    
--    local loss12 = 0   
    
--    local subIterator = opt.batchSize/subBatchSize
--    if batch_start_idx + opt.batchSize > trainData.data:size(1) then
--	subIterator = math.ceil((trainData.data:size(1)-batch_start_idx+1)/subBatchSize)
--    end
--    for interIter = 1, subIterator do
         local _, loss_12 = optMethod(feval, params, optState)
   	 batch_start_idx = batch_start_idx + opt.batchSize

--         if iter%400 == 0 then
	if batch_start_idx > trainData.data:size(1) then
		randPerm = torch.randperm(trainData.data:size(1)):long()
		batch_start_idx = 1
        end
--       tmp = tmp + loss_12[1]
--       loss12 = loss12 + loss_12[1]
 --   end
--	tmp = tmp + loss12
    model:evaluate()

    tmp = tmp+loss_12[1]
    if iter%400 == 0 then
      tmp = tmp/400
      losses[#losses + 1] = tmp
      tmp = 0
	print("testing confusion Matrix ===============")

      -- testing error
      rndIDX = torch.randperm(testData.data:size(1)):long()
      tmpErr=0
      local predictions = torch.Tensor(testData.labels:size(1)):fill(0)
      for i = 1,1000 do
	local tmppred = (model:forward(testData.data[{ {(i-1)*10+1,i*10},{},{},{} }]:cuda())):float()
	local _,tmpout = torch.max(tmppred,2)
	tmpout = (tmpout:reshape(tmpout:size(1))):float()
	predictions[{ {(i-1)*10+1, i*10} }] = tmpout
	tmpErr = tmpErr + torch.sum(tmpout:eq((testData.labels[{ {(i-1)*10+1, i*10} }]):float()))/10
      end
      confusionMatrix(predictions, testData.labels, {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'})

      tsError[#tsError + 1] = 100 - (tmpErr/1000*100)
--	 print(string.format("testing error : %.2f", tsError[#tsError]))
 
	print("training confusion Matrix ===============")
     
       -- training error
      tmpErr=0
      predictions = torch.Tensor(trainData.labels:size(1)):fill(0)
      for i = 1,trainData.data:size(1)/10 do
        local tmppred = (model:forward(trainData.data[{ {(i-1)*10+1,i*10},{},{},{} }]:cuda())):float()
        local _,tmpout = torch.max(tmppred,2)
        tmpout = (tmpout:reshape(tmpout:size(1))):float()
        predictions[{ {(i-1)*10+1, i*10} }] = tmpout
        tmpErr = tmpErr + torch.sum(tmpout:eq((trainData.labels[{ {(i-1)*10+1, i*10} }]):float()))/10
      end
      confusionMatrix(predictions, trainData.labels, {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'})
      trError[#trError + 1] = 100 - (tmpErr/(trainData.data:size(1)/10)*100)



--      model:clearState() 
--      torch.save("model/" .. numLayer .. "/model_" .. os.date('%c') .. ".dat", model)
    end

    if iter == 400 then
	optState.learningRate = optState.learningRate*10
    end

    if (iter%100) == 0 then
--	print("===================================================================================================")
--      print(string.format("iteration : %6s / %6s, ==> loss = %6.6f   test error = %.2f   learningRate = %.6f", iter, numIter, loss_12[1], tsError[#tsError], optState.learningRate))
	print(string.format("iteration : %6s / %6s, ==> loss = %6.6f\tlearningRate = %.6f", iter, numIter, loss_12[1], optState.learningRate))
        if iter%400 == 0 then
	   print(string.format("training error = %2.2f\ttesting error = %2.2f", trError[#trError], tsError[#tsError]))
	end
--	print("===================================================================================================")
--	print("\n")      
    end
  end
  
  return model, losses, tsError, trError

end


