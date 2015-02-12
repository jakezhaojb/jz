local SpatialMlpUnPooling, parent = torch.class('jz.SpatialMlpUnPooling', 'nn.Module')

function SpatialMlpUnPooling:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH)
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

function SpatialMlpUnPooling:reset(stdv)
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

function SpatialMlpUnPooling:updateOutput(input)
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
      self.output = torch.Tensor():cuda()
      jz.SpatialMlpUnPooling_updateOutput(self, input)
   else
      local kW = self.kW
      local kH = self.kH
      self.output = torch.Tensor():resize(input:size(1), input:size(2), kH*input:size(3), kW*input:size(4)):typeAs(input)
      local nBatches = inputSize[1]
      local nOutputPlane = inputSize[2]
      local nOutputCols = inputSize[4] * kW
      local nOutputRows = inputSize[3] * kH
      local nCols = inputSize[4]
      local nRows = inputSize[3]

      for batch = 1, nBatches do
         for plane = 1, nOutputPlane do
            local input_elem = input[batch][plane]
            local weight_elem = self.weight[plane]
            for i = 1, nRows do
               for j = 1, nCols do
                  self.output[{ batch,plane,{(i-1)*kH+1,i*kH},{(j-1)*kW+1,j*kW}} ] = torch.mul(weight_elem, input_elem[i][j])
               end
            end
         end -- end for plane
      end -- end for batch
   end -- end if
   return self.output
end

function SpatialMlpUnPooling:updateGradInput(input, gradOutput)
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
      jz.SpatialMlpUnPooling_updateGradInput(self, input, gradOutput)
   else
      self.gradInput = torch.Tensor():resizeAs(input):typeAs(input):fill(0)
      local kW = self.kW
      local kH = self.kH
      local nBatches = inputSize[1]
      local nOutputPlane = inputSize[2]
      local nOutputCols = inputSize[4] * kW
      local nOutputRows = inputSize[3] * kH
      local nCols = inputSize[4]
      local nRows = inputSize[3]

      for batch = 1, nBatches do
         for plane = 1, nOutputPlane do
            local input_elem = input[batch][plane]
            local weight_elem = self.weight[plane]
            for i = 1, nRows do
               for j = 1, nCols do
                  self.gradInput[batch][plane][i][j] = torch.cmul(weight_elem, gradOutput[{ batch,plane,{(i-1)*kH+1,i*kH},{(j-1)*kW+1,j*kW}} ]):sum()
               end
            end
         end
      end
   end
   return self.gradInput
end


function SpatialMlpUnPooling:accGradParameters(input, gradOutput, scale)
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
      jz.SpatialMlpUnPooling_accGradParameters(self, input, gradOutput, scale)
   else
      local kW = self.kW
      local kH = self.kH
      local nBatches = inputSize[1]
      local nOutputPlane = inputSize[2]
      local nOutputCols = inputSize[4] * kW
      local nOutputRows = inputSize[3] * kH
      local nCols = inputSize[4]
      local nRows = inputSize[3]
      for plane = 1, nOutputPlane do
         gradWeight_elem = self.gradWeight[plane]
         for batch = 1, nBatches do
            local input_elem = input[batch][plane]
            local gradOutput_elem = gradOutput[batch][plane]
            for i = 1, nRows do
               for j = 1, nCols do
                  gradWeight_elem:add(gradOutput_elem[{ {(i-1)*kH+1,i*kH},{(j-1)*kW+1,j*kW} }]*input_elem[i][j]*scale)
               end
            end
         end
      end
   end

end
