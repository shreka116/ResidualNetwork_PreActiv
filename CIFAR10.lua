require 'paths'
require 'torch'
function CIFAR10()
  if not paths.filep('../dataset/CIFAR10_trainData.t7') then 
--[[
----------------------------------------------------------------------
-- This script downloads and loads the CIFAR-10 dataset
-- http://www.cs.toronto.edu/~kriz/cifar.html
----------------------------------------------------------------------
  print '==> downloading dataset'

-- Here we download dataset files. 

-- Note: files were converted from their original format
-- to Torch's internal format.

-- The CIFAR-10 dataset provides  3 files:
--    + train: training data
--    + test:  test data

  local tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar10.t7.tgz'

  if not paths.dirp('../dataset/cifar-10-batches-t7') then
    os.execute('wget ' .. '-P ~/Desktop/DLstudy/ResidualNet/dataset/ ' .. tar)
    os.execute('tar xf ' .. '~/Desktop/DLstudy/ResidualNet/dataset/' .. paths.basename(tar) .. ' -C ~/Desktop/DLstudy/ResidualNet/dataset/')
  end
----------------------------------------------------------------------
]]--

    print '==> generatig dataset'

-- We load the dataset from disk, it's straightforward

    local trsize = 50000
--    local vlsize = 5000
    local tesize = 10000

    local tdata = {
      data = torch.Tensor(trsize, 3*32*32),
      labels = torch.Tensor(trsize),
      size = function() return trsize end
    }

    local trainData = {
      data = torch.Tensor(trsize, 3*32*32),
      labels = torch.Tensor(trsize),
      size = function() return trsize end
    }
    for i = 0,4 do
      local subset = torch.load('../dataset/cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
      tdata.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
      tdata.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
    end
    tdata.labels = tdata.labels + 1

    local subset = torch.load('../dataset/cifar-10-batches-t7/test_batch.t7', 'ascii')
    local testData = {
      data = subset.data:t():double(),
      labels = subset.labels[1]:double(),
      size = function() return tesize end
    }
    testData.labels = testData.labels + 1
    
 --   local validData = {
 --     data = torch.Tensor(vlsize, 3*32*32),
  --    labels = torch.Tensor(vlsize),
   --   size = function() return vlsize end
  --  }
    
    
-- resize dataset (if using small version)
    trainData.data = tdata.data[{ {1,trsize} }]
    trainData.labels = tdata.labels[{ {1,trsize} }]

 --   validData.data = tdata.data[{ {trsize+1, trsize+vlsize} }]
 --   validData.labels = tdata.labels[{ {trsize+1, trsize+vlsize} }]

    testData.data = testData.data[{ {1,tesize} }]
    testData.labels = testData.labels[{ {1,tesize} }]

-- reshape data                                                                                     
    trainData.data = trainData.data:reshape(trsize,3,32,32)
--    validData.data = validData.data:reshape(vlsize,3,32,32)
    testData.data = testData.data:reshape(tesize,3,32,32)
    
	local pixMean_1 = torch.repeatTensor(torch.mean(trainData.data[{ {},{1},{},{} }], 1),trainData.data:size(1),1,1,1)
	local pixMean_2 = torch.repeatTensor(torch.mean(trainData.data[{ {},{2},{},{} }], 1),trainData.data:size(1),1,1,1)
	local pixMean_3 = torch.repeatTensor(torch.mean(trainData.data[{ {},{3},{},{} }], 1),trainData.data:size(1),1,1,1)

	local pixSTD_1  = torch.repeatTensor(torch.std(trainData.data[{ {},{1},{},{} }], 1), trainData.data:size(1),1,1,1)
	local pixSTD_2  = torch.repeatTensor(torch.std(trainData.data[{ {},{2},{},{} }], 1), trainData.data:size(1),1,1,1)
	local pixSTD_3  = torch.repeatTensor(torch.std(trainData.data[{ {},{3},{},{} }], 1), trainData.data:size(1),1,1,1)

    trainData.data[{ {},{1},{},{} }]:csub(pixMean_1):cdiv(pixSTD_1)
    trainData.data[{ {},{2},{},{} }]:csub(pixMean_2):cdiv(pixSTD_2)
    trainData.data[{ {},{3},{},{} }]:csub(pixMean_3):cdiv(pixSTD_3)        
    
--    validData.data[{ {},{1},{},{} }]:csub(torch.repeatTensor(torch.mean(validData.data[{ {},{1},{},{} }], 1),validData.data:size(1),1,1,1))
--    validData.data[{ {},{2},{},{} }]:csub(torch.repeatTensor(torch.mean(validData.data[{ {},{2},{},{} }], 1),validData.data:size(1),1,1,1))
--    validData.data[{ {},{3},{},{} }]:csub(torch.repeatTensor(torch.mean(validData.data[{ {},{3},{},{} }], 1),validData.data:size(1),1,1,1))        

	pixMean_1 = torch.repeatTensor(torch.mean(testData.data[{ {},{1},{},{} }], 1),testData.data:size(1),1,1,1)
        pixMean_2 = torch.repeatTensor(torch.mean(testData.data[{ {},{2},{},{} }], 1),testData.data:size(1),1,1,1)
        pixMean_3 = torch.repeatTensor(torch.mean(testData.data[{ {},{3},{},{} }], 1),testData.data:size(1),1,1,1)

        pixSTD_1  = torch.repeatTensor(torch.std(testData.data[{ {},{1},{},{} }], 1),testData.data:size(1),1,1,1)
        pixSTD_2  = torch.repeatTensor(torch.std(testData.data[{ {},{2},{},{} }], 1),testData.data:size(1),1,1,1)
        pixSTD_3  = torch.repeatTensor(torch.std(testData.data[{ {},{3},{},{} }], 1),testData.data:size(1),1,1,1)
    

    testData.data[{ {},{1},{},{} }]:csub(pixMean_1):cdiv(pixSTD_1)
    testData.data[{ {},{2},{},{} }]:csub(pixMean_2):cdiv(pixSTD_2)
    testData.data[{ {},{3},{},{} }]:csub(pixMean_3):cdiv(pixSTD_3)  
 
 --[[
    trainData.data[{ {},{1},{},{} }]:csub(torch.repeatTensor(torch.sum(trainData.data[{ {},{1},{},{} }],1)/trainData.data:size(1),trainData.data:size(1),1,1,1))
    trainData.data[{ {},{2},{},{} }]:csub(torch.repeatTensor(torch.sum(trainData.data[{ {},{2},{},{} }],1)/trainData.data:size(1),trainData.data:size(1),1,1,1))
    trainData.data[{ {},{3},{},{} }]:csub(torch.repeatTensor(torch.sum(trainData.data[{ {},{3},{},{} }],1)/trainData.data:size(1),trainData.data:size(1),1,1,1))
    
    testData.data[{ {},{1},{},{} }]:csub(torch.repeatTensor(torch.sum(testData.data[{ {},{1},{},{} }],1)/testData.data:size(1),testData.data:size(1),1,1,1))
    testData.data[{ {},{2},{},{} }]:csub(torch.repeatTensor(torch.sum(testData.data[{ {},{2},{},{} }],1)/testData.data:size(1),testData.data:size(1),1,1,1))
    testData.data[{ {},{3},{},{} }]:csub(torch.repeatTensor(torch.sum(testData.data[{ {},{3},{},{} }],1)/testData.data:size(1),testData.data:size(1),1,1,1))
    ]]--
    
    torch.save("../dataset/CIFAR10_trainData.t7", trainData)
 --   torch.save("../dataset/CIFAR10_validData.t7", validData)
    torch.save("../dataset/CIFAR10_testData.t7", testData)
    return trainData, testData
  else
    print '==> loading dataset'

    local trainData = torch.load("../dataset/CIFAR10_trainData.t7")
 --   local validData = torch.load("../dataset/CIFAR10_validData.t7")
    local testData  = torch.load("../dataset/CIFAR10_testData.t7")
    return trainData, testData
  end

--[[
----------------------------------------------------------------------
print '==> visualizing data'

-- Visualization is quite easy, using itorch.image().
if itorch then
   print('training data:')
   itorch.image(trainData.data[{ {1,256} }])
   print('test data:')
   itorch.image(testData.data[{ {1,256} }])
end
]]--
end
