require 'cutorch'
require 'cunn'
require 'nn'
require 'optim'
require 'cudnn'

require 'gnuplot'
require 'paths'
require 'image'
require 'math'

require 'loadCIFAR10'
require 'trainBottleNeck_new'
torch.setnumthreads(2)
local cudaOption = true

--------------------------------------------------------
-- load dataset
--------------------------------------------------------
local trainData, testData = loadCIFAR10('../dataset')
--------------------------------------------------------
-- training Plain and ResNet
--------------------------------------------------------


local numLayer_1 = 56

local model_BottleNeck, loss_BottleNeck, testErr_BottleNeck, trainErr_BottleNeck = trainBottleNeck_new(trainData, testData, numLayer_1)
model_BottleNeck:clearState()
torch.save('model_BottleNeck_final.dat', model_BottleNeck)
torch.save('testErr_BottleNeck.t7', testErr_BottleNeck)
torch.save('trainErr_BottleNeck.t7', trainErr_BottleNeck)
torch.save('loss_BottleNeck.t7', loss_BottleNeck)
