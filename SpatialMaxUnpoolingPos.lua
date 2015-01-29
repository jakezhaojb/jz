local SpatialMaxUnpoolingPos, parent = torch.class('jz.SpatialMaxUnpoolingPos', 'nn.Module')

function SpatialMaxUnpoolingPos:__init(kW, kH)
   parent.__init(self)
   self.kW = kW
   self.kH = kH
end

function SpatialMaxUnpoolingPos:updateOutput(input)
   if input:type() == 'torch.CudaTensor' then
      -- TODO you have to slice it into three before inputting into CUDA file, to reduce the pain...
      --TODO
      --[[
      self.output_p = torch.CudaTensor()
      self.output_dx = torch.CudaTensor()
      self.output_dy = torch.CudaTensor()
      jz.SpatialMaxUnpoolingPos_updateOutput(self, input)
      join_table = nn.JoinTable(2)
      join_table:cuda()
      self.output = join_table:forward({self.output_p, self.output_dx, self.output_dy})
      --]]
   else
      local inputSize = input:size()
      if inputSize:size() ~= 4 then
         print('Expected a 4D Tensor in SpatialMaxPooling')
         return nil
      end
      local kW = self.kW
      local kH = self.kH
      local nBatches = inputSize[1]
      assert(inputSize[2] % 3 == 0)
      local nOutputPlanes = inputSize[2] / 3
      local nOutputCols = inputSize[4]*kW -- TODO, in case of not mod?
      local nOutputRows = inputSize[3]*kH
      --local maxW = nOutputCols / kW
      --local maxH = nOutputRows / kH

      self.output:resize(inputSize[1], nOutputPlanes, nOutputRows, nOutputCols):typeAs(input):fill(0)
      local input_p = input[{{}, {1,nOutputPlanes},{},{} }]
      local input_dx = input[{{}, {nOutputPlanes+1,2*nOutputPlanes},{},{} }]
      local input_dy = input[{{}, {2*nOutputPlanes+1,3*nOutputPlanes},{},{} }]

      for batch = 1,nBatches do
         for inplane = 1, nOutputPlanes do
            local dwPlane = input_dx[batch][inplane]
            local dhPlane = input_dy[batch][inplane]
            local pPlane = input_p[batch][inplane]
            local oi = 1
            for i = 1, nOutputCols, kH do
               local oj = 1 
               for j = 1, nOutputRows, kW do
                  self.output[batch][inplane][i+dhPlane[oi][oj]-1][j+dwPlane[oi][oj]-1] = pPlane[oi][oj]
                  oj = oj + 1
               end
               oi = oi + 1
            end
         end
      end
   end
   collectgarbage()
   return self.output
end


function SpatialMaxUnpoolingPos:updateGradInput(input, gradOutput)
   local inputSize = input:size()
   local nOutputPlanes = inputSize[2] / 3
   if input:type() == 'torch.CudaTensor' then
      -- TODO
      --jz.SpatialMaxUnpoolingPos(self, input, gradOutput)
   else
      self.gradInput = torch.Tensor():resizeAs(input):fill(0):typeAs(input)
      local kW = self.kW
      local kH = self.kH
      local nBatches = inputSize[1]

      local nInputCols = inputSize[4]
      local nInputRows = inputSize[3]
      local nOutputCols = inputSize[4]*kW
      local nOutputRows = inputSize[3]*kH

      local gradInput_p = torch.Tensor(inputSize[1], nOutputPlanes, nInputRows, nInputCols):typeAs(input)
      local gradInput_dx = torch.Tensor(inputSize[1], nOutputPlanes, nInputRows, nInputCols):typeAs(input):fill(0)
      local gradInput_dy = torch.Tensor(inputSize[1], nOutputPlanes, nInputRows, nInputCols):typeAs(input):fill(0)
      local input_dx = input[{{}, {nOutputPlanes+1,2*nOutputPlanes},{},{} }]
      local input_dy = input[{{}, {2*nOutputPlanes+1,3*nOutputPlanes},{},{} }]

      for batch = 1, nBatches do
         for inplane = 1, nOutputPlanes do
            local dwPlane = input_dy[batch][inplane]
            local dhPlane = input_dx[batch][inplane]
            local oi = 1
            for i = 1, nOutputRows, kH do
               local oj = 1
               for j = 1, nOutputCols, kW do 
                  gradInput_p[batch][inplane][oi][oj] = gradOutput[batch][inplane][i+dhPlane[oi][oj]-1][j+dwPlane[oi][oj]-1]
                  gradInput_dx[batch][inplane][oi][oj] = 0 
                  gradInput_dy[batch][inplane][oi][oj] = 0
                  oj = oj + 1
               end
               oi = oi + 1
            end
         end
      end
      join_table = nn.JoinTable(2)
      self.gradInput = join_table:forward({gradInput_p, gradInput_dx, gradInput_dy})
      collectgarbage()
      return self.gradInput
   end
end
