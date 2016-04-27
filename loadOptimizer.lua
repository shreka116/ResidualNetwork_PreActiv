function loadOptimizer(types)

  local opt = {
    optimizer = 'sgd',
    batchSize = 128,
    maxIter   = 64000
  }

  local optimizerState, optimizerMethod
  
  if types == "110" then
    if opt.optimizer == 'sgd' then
      optimizerState = {
        learningRate = 1e-2,
        weightDecay = 1e-4,
        momentum = 0.9,
        nesterov = true,
        dampening= 0
      }
      optimizerMethod = optim.sgd
    elseif opt.optimizer == 'adagrad' then
      optimizerState = {
        learningRate = 1e-1
      }
      optimizerMethod = optim.adagrad
    elseif opt.optimizer == 'L-BFGS' then
      optimizerState = {
        learningRate = 1e-3,
        maxIter = 3,
        nCorrection = 10
      }
      optimizerMethod = optim.lbfgs
    end
  elseif types == "164" then
    if opt.optimizer == 'sgd' then
      optimizerState = {
        learningRate = 1e-2,
        weightDecay = 1e-4,
        momentum = 0.9,
        nesterov = true,
        dampening= 0
      }
      optimizerMethod = optim.sgd
    elseif opt.optimizer == 'adagrad' then
      optimizerState = {
        learningRate = 1e-1
      }
      optimizerMethod = optim.adagrad
    elseif opt.optimizer == 'L-BFGS' then
      optimizerState = {
        learningRate = 1e-3,
        maxIter = 3,
        nCorrection = 10
      }
      optimizerMethod = optim.lbfgs
    end
  end


  return opt, optimizerState, optimizerMethod

end
