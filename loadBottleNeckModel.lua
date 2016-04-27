require 'addBottleNeckLayer'
require 'cudnn'
local nninit = require 'nninit'

function loadBottleNeckModel(numLayer)
-- nn version
--  local model = nn.Sequential()
--	model:add(nn.SpatialConvolution(3,8,3,3,1,1,1,1))
--  model:add(nn.SpatialBatchNormalization(16))
--  model:add(nn.ReLU())

-- cudnn version
  local model = nn.Sequential()
  model:add(nn.SpatialConvolution(3,16,3,3,1,1,1,1):init('weight', nninit.kaiming, {gain='relu'}):init('bias', nninit.constant, 0))
--  model:add(nn.SpatialBatchNormalization(16))
--  model:add(nn.ReLU())

  local numBlock = (numLayer-2)/3
  --model = addResLayer(model, 8)
  
  model = addBottleNeckLayer(model, 16, 64, 1) 
 
  for i = 1, (numBlock/3)-1 do
    model = addBottleNeckLayer(model, 64)
  end
  
  model = addBottleNeckLayer(model, 64, 128, 2)
  
  
  for i = 1, (numBlock/3)-1 do
    model = addBottleNeckLayer(model, 128)
  end
  
  model = addBottleNeckLayer(model, 128, 256, 2)
  
  for i = 1, (numBlock/3)-1 do
    model = addBottleNeckLayer(model, 256)
  end
  model:add(nn.SpatialBatchNormalization(256))
  model:add(nn.ReLU())
  model:add(nn.SpatialAveragePooling(8,8))
  model:add(nn.Reshape(256))
  model:add(nn.Linear(256,10))
  model:add(nn.LogSoftMax())

 
  
  return model
end
