local Predictions = { output = nil; predictions = {} }

function Predictions:new(o)
  o = o or {}
  setmetatable(o, self)
  self.__index = self
  print(o)
  return o
end

function Predictions:init()
  local label = require 'overfeat_label'
  for i=1,self.output:size()[1] do
    table.insert(self.predictions, {category_id=i, label=label[i], confidence=self.output[i]})
  end
end

-- output needs to be from output = net:forward(input)
function Predictions:sort()
  table.sort(self.predictions, function(a,b) return a.confidence > b.confidence end)
end

function Predictions:getN(limit)
  local t={};
  for i=1,limit do
    t[i] = self.predictions[i]
  end
  return t
end

return Predictions
