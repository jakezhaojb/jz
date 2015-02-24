local SpatialMlpPoolingWTA, parent = torch.class('jz.SpatialMlpPoolingWTA', 'nn.Module')

function SpatialMlpPoolingWTA:__init(nInputPlane, nOutputPlane, kW, kH)
   parent.__init(self)

   assert(nInputPlane == nOutputPlane)

   -- TODO 
   dW = dW or kW
   dH = dH or kH

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.weight = torch.Tensor(nOutputPlane, kH, kW)
   --self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, kH, kW)
   --self.gradBias = torch.Tensor(nOutputPlane)

   self:reset()
end


function SpatialMlpPoolingWTA:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv) 
      end)
      --self.bias:apply(function()
      --   return torch.uniform(-stdv, stdv)
      --end)
   else
      self.weight:uniform(-stdv, stdv)
      --self.bias:uniform(-stdv, stdv)
   end
end


function SpatialMlpPoolingWTA:updateOutput(input)
   if input:dim() ~= 4 then
      if input:dim() == 3 then
         input = input:resize(1, input:size(1), input:size(2), input:size(3)):typeAs(input)
      else
         print('Expected a 3D/4D Tensor in SpatialMlpUnPooling')
         return nil
      end
   end
   -- Initialize
   local inputSize = input:size()
   local kW = self.kW
   local kH = self.kH
   local nBatches = inputSize[1]
   local nOutputPlane = inputSize[2]
   local nOutputCols = math.floor(inputSize[4] / kW)
   local nOutputRows = math.floor(inputSize[3] / kH)
   local maxW = nOutputCols * kW
   local maxH = nOutputRows * kH
   local nCols = inputSize[4]
   local nRows = inputSize[3]
   if input:type() == 'torch.CudaTensor' then
      self.output = torch.Tensor():cuda()
      self.dx = torch.Tensor():cuda()
      self.dy = torch.Tensor():cuda()
      jz.SpatialMlpPoolingWTA_updateOutput(self, input)
   else
      self.output = torch.Tensor():resize(input:size(1), input:size(2), nOutputRows, nOutputCols):typeAs(input)
      self.dx = torch.Tensor():resizeAs(self.output):typeAs(self.output):fill(0)
      self.dy = torch.Tensor():resizeAs(self.output):typeAs(self.output):fill(0)
      for batch = 1, nBatches do
         for plane = 1, nOutputPlane do
            local oi = 1
            local input_elem = input[batch][plane]
            local weight_elem = self.weight[plane]
            for i = 1, maxH, kH do
               local oj = 1
               for j = 1, maxW, kW do
                  local pooledVal = 0
                  dx, dy = 0, 0
                  for h = 1, kH do
                     for w = 1, kW do
                        pooledVal_tmp = input_elem[i+h-1][j+w-1] * weight_elem[h][w]
                        if pooledVal_tmp > pooledVal  then
                           pooledVal = pooledVal_tmp
                           dx = w
                           dy = h
                        end
                     end
                  end
                  self.output[batch][plane][oi][oj] = pooledVal
                  self.dx[batch][plane][oi][oj] = dx
                  self.dy[batch][plane][oi][oj] = dy
                  oj = oj + 1
               end
               oi = oi + 1
            end
         end
      end
   end
   return self.output
end


function SpatialMlpPoolingWTA:updateGradInput(input, gradOutput)
   if input:dim() ~= 4 then
      if input:dim() == 3 then
         input = input:resize(1, input:size(1), input:size(2), input:size(3)):typeAs(input)
      else
         print('Expected a 3D/4D Tensor in SpatialMlpUnPooling')
         return nil
      end
   end
   inputSize = input:size()
   if input:type() == 'torch.CudaTensor' then
      self.gradInput = torch.Tensor():cuda()
      jz.SpatialMlpPoolingWTA_updateGradInput(self, input, gradOutput)
   else
      self.gradInput = torch.Tensor():resizeAs(input):typeAs(input):fill(0)
      local kW = self.kW
      local kH = self.kH
      local nBatches = inputSize[1]
      local nOutputPlane = inputSize[2]
      local nOutputCols = math.floor(inputSize[4]/kW)
      local nOutputRows = math.floor(inputSize[3]/kH)
      local maxW = nOutputCols * kW
      local maxH = nOutputRows * kH
      local nCols = inputSize[4]
      local nRows = inputSize[3]

      for batch = 1, nBatches do
         for plane = 1, nOutputPlane do
            local gradOutput_elem = gradOutput[batch][plane]
            local weight_elem = self.weight[plane]
            local dx_elem = self.dx[batch][plane]
            local dy_elem = self.dy[batch][plane]
            local oi = 1
            for i = 1, maxH, kH do
               local oj = 1
               for j = 1, maxW, kW do
                  self.gradInput[batch][plane][i+dy_elem[oi][oj]-1][j+dx_elem[oi][oj]-1] = gradOutput_elem[oi][oj]*weight_elem[dy_elem[oi][oj]][dx_elem[oi][oj]]
                  oj = oj + 1
               end
               oi = oi + 1
            end
         end
      end
   end
   return self.gradInput
end
            

function SpatialMlpPoolingWTA:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   if input:dim() ~= 4 then
      if input:dim() == 3 then
         input = input:resize(1, input:size(1), input:size(2), input:size(3)):typeAs(input)
      else
         print('Expected a 3D/4D Tensor in SpatialMlpUnPooling')
         return nil
      end
   end
   inputSize = input:size()

   if input:type() == 'torch.CudaTensor' then
      jz.SpatialMlpPoolingWTA_accGradParameters(self, input, gradOutput, scale)
   else
      local kW = self.kW
      local kH = self.kH
      local nBatches = inputSize[1]
      local nOutputPlane = inputSize[2]
      local nOutputCols = math.floor(inputSize[4] / kW)
      local nOutputRows = math.floor(inputSize[3] / kH)
      local maxW = nOutputCols * kW
      local maxH = nOutputRows * kH
      local nCols = inputSize[4]
      local nRows = inputSize[3]

      for plane = 1, nOutputPlane do
         gradWeight_elem = self.gradWeight[plane]
         for batch = 1, nBatches do
            local input_elem = input[batch][plane]
            local gradOutput_elem = gradOutput[batch][plane]
            local dx_elem = self.dx[batch][plane]
            local dy_elem = self.dy[batch][plane]
            for i = 1, nOutputRows do
               for j = 1, nOutputCols do
                  local gradWeight_elem_tmp = input_elem[{ {(i-1)*kH+1,i*kH},{(j-1)*kW+1,j*kW} }]*gradOutput_elem[i][j]*scale
                  local gradWeight_elem_WTAbuffer = torch.Tensor():resizeAs(gradWeight_elem_tmp):typeAs(gradWeight_elem_tmp):fill(0)
                  gradWeight_elem_WTAbuffer[{ dy_elem[i][j], dx_elem[i][j] }] = gradWeight_elem_tmp[{ dy_elem[i][j], dx_elem[i][j] }]
                  gradWeight_elem:add(gradWeight_elem_WTAbuffer)
               end
            end
         end
      end
   end
end
