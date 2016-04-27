require 'cudnn'
local nninit = require 'nninit'
function addBottleNeckLayer(model, inChannel, outChannel, stride, first)
  
  -- defaults setting
  if outChannel == nil then
    outChannel = inChannel
  end
  if stride == nil then
    stride = 1
  end
  
  -- residual model
  local subSeq = nn.Sequential()
  subSeq:add(nn.SpatialBatchNormalization(inChannel):init('weight', nninit.normal, 1.0, 0.002):init('bias', nninit.constant, 0))
  subSeq:add(nn.ReLU())
  subSeq:add(nn.SpatialConvolution(inChannel, outChannel/4, 1, 1, stride, stride, 0, 0):init('weight', nninit.kaiming, {gain='relu'}):init('bias', nninit.constant, 0))
--  subSeq:add(nn.SpatialBatchNormalization(outChannel))
  subSeq:add(nn.SpatialBatchNormalization(outChannel):init('weight', nninit.normal, 1.0, 0.002):init('bias', nninit.constant, 0))
  subSeq:add(nn.ReLU())
  subSeq:add(nn.SpatialConvolution(outChannel/4, outChannel/4, 3, 3, 1, 1, 1, 1):init('weight', nninit.kaiming, {gain='relu'}):init('bias', nninit.constant, 0))
--  subSeq:add(nn.SpatialBatchNormalization(outChannel))
  subSeq:add(nn.SpatialBatchNormalization(outChannel):init('weight', nninit.normal, 1.0, 0.002):init('bias', nninit.constant, 0))
  subSeq:add(nn.ReLU())
  subSeq:add(nn.SpatialConvolution(outChannel/4, outChannel, 1, 1, 1, 1, 0, 0):init('weight', nninit.kaiming, {gain='relu'}):init('bias', nninit.constant, 0)) 



  -- skip connection
  local skip = nn.ConcatTable()
  local skipSeq = nn.Sequential()
  
  skip:add(subSeq)
  
  if outChannel > inChannel then
--    -- option A
--    skipSeq:add(nn.SpatialAveragePooling(1, 1, stride, stride))
--    skipSeq:add(nn.Padding(1, (outChannel-inChannel), 3))

  -- option B
--    skipSeq:add(nn.SpatialAveragePooling(1, 1, stride, stride))
    skipSeq:add(nn.SpatialBatchNormalization(inChannel))
    skipSeq:add(nn.ReLU())
    skipSeq:add(nn.SpatialConvolution(inChannel, outChannel, 1, 1, stride, stride, 0, 0):init('weight', nninit.kaiming, {gain='relu'}):init('bias', nninit.constant, 0))
  else
    skipSeq:add(nn.Identity())
    
 --   skip:add(nn.Identity())
  end
  skip:add(skipSeq)
  
  model:add(skip)
  model:add(nn.CAddTable())
  -- add two outputs

--  model:add(nn.ReLU())
  
  
  return model
  
end
